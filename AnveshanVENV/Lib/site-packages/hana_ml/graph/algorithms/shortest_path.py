# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from .algorithm_base import AlgorithmBase
from .. import Graph
from ..constants import DEFAULT_DIRECTION, DIRECTIONS


class ShortestPath(AlgorithmBase):
    """
    Given a source and target vertex_key with optional weight and direction,
    get the shortest path between them.

    The procedure may fail for HANA versions prior to SP05 therefore this
    is checked at execution time.

    The user can take the results and visualize them with libraries
    such as networkX using the :func:`~edges` property.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> sp = hga.ShortestPath(graph=g).execute(source="1", target="3")
    >>>
    >>> print("Vertices", sp.vertices)
    >>> print("Edges", sp.edges)
    >>> print("Weight:", sp.weight)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
                        DO (
                            IN i_startVertex {vertex_dtype} => '{start_vertex}',
                            IN i_endVertex {vertex_dtype} => '{end_vertex}',
                            IN i_direction NVARCHAR(10) => '{direction}',
                            OUT o_vertices TABLE ({vertex_columns}, "VERTEX_ORDER" BIGINT) => ?,
                            OUT o_edges TABLE ({edge_columns}, "EDGE_ORDER" BIGINT) => ?,
                            OUT o_scalars TABLE ("WEIGHT" DOUBLE) => ?
                        )
                        LANGUAGE GRAPH
                        BEGIN
                            GRAPH g = Graph("{schema}", "{workspace}");
                            VERTEX v_start = Vertex(:g, :i_startVertex);
                            VERTEX v_end = Vertex(:g, :i_endVertex);

                            {weighted_definition}

                            o_vertices = SELECT {vertex_select}, :VERTEX_ORDER FOREACH v 
                                         IN Vertices(:p) WITH ORDINALITY AS VERTEX_ORDER;
                            o_edges = SELECT {edge_select}, :EDGE_ORDER FOREACH e 
                                      IN Edges(:p) WITH ORDINALITY AS EDGE_ORDER;

                            DOUBLE p_weight= DOUBLE(WEIGHT(:p));
                            o_scalars."WEIGHT"[1L] = :p_weight;
                        END;
                       """

        self._graph_script_vars = {
            "start_vertex": ("source", True, None, None),
            "end_vertex": ("target", True, None, None),
            "weight": ("weight", False, str, None),
            "direction": ("direction", False, str, DEFAULT_DIRECTION),
            "schema": (None, False, str, self._graph.workspace_schema),
            "workspace": (None, False, str, self._graph.workspace_name),
            "vertex_dtype": (None, False, str, self._graph.vertex_key_col_dtype),
            "vertex_columns": (None, False, str, self._default_vertex_cols()),
            "edge_columns": (None, False, str, self._default_edge_cols()),
            "vertex_select": (None, False, str, self._default_vertex_select("v")),
            "edge_select": (None, False, str, self._default_edge_select("e")),
        }

    def _process_parameters(self, arguments):
        super()._process_parameters(arguments)

        # Construct the WeightedPath part of the SQL Statement depending
        # on if a weight parameter was provided or not
        if self._templ_vals["weight"]:
            self._templ_vals[
                "weighted_definition"
            ] = """
                    WeightedPath<DOUBLE> p = Shortest_Path(:g, :v_start,
                    :v_end, (Edge e) => DOUBLE{{ return DOUBLE(:e."{weight}"); }},
                    :i_direction);
                """.format(weight=self._templ_vals["weight"])
        else:
            self._templ_vals[
                "weighted_definition"
            ] = """
                    WeightedPath<BIGINT> p = Shortest_Path(:g, :v_start,
                    :v_end, :i_direction);
                """

    def _validate_parameters(self):
        # Version check
        if int(self._graph.connection_context.hana_major_version()) < 4:
            raise EnvironmentError(
                "SAP HANA version is not compatible with this method"
            )

        # Check Direction
        if self._templ_vals["direction"] not in DIRECTIONS:
            raise KeyError(
                "Direction needs to be one of {}".format(", ".join(DIRECTIONS))
            )

        # Check if the vertices exist
        if not self._graph.has_vertices([self._templ_vals["start_vertex"]]):
            raise ValueError(
                "Start vertex '{}' is not part of the graph".format(
                    self._templ_vals["start_vertex"]
                )
            )

        if not self._graph.has_vertices([self._templ_vals["end_vertex"]]):
            raise ValueError(
                "Target vertex '{}' is not part of the graph".format(
                    self._templ_vals["end_vertex"]
                )
            )

    def execute(  # pylint: disable=arguments-differ
            self,
            source: str,
            target: str,
            weight: str = None,
            direction: str = DEFAULT_DIRECTION,
    ) -> "ShortestPath":
        """
        Executes the calculation of the shortest path.

        Parameters
        ----------
        source : str
            Vertex key from which the shortest path will start.
        target : str
            Vertex key from which the shortest path will end.
        weight : str, optional
            Variable for column name to which to apply the weight.

            Defaults to None.
        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm results.

            Defaults to OUTGOING.

        Returns
        -------
        ShortestPath
            ShortestPath object instance
        """
        return super(ShortestPath, self).execute(
            source=source, target=target, weight=weight, direction=direction
        )

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the vertices of the shortest
            path
        """
        vertex_cols = [
            col
            for col in self._graph.vertices_hdf.columns
            if col in self._default_vertices_column_filter
        ]
        vertex_cols.append("VERTEX_ORDER")

        return DataFrame(self._results.get("o_vertices", None), columns=vertex_cols)

    @property
    def edges(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the edges of the shortest path
        """
        edge_cols = [
            col
            for col in self._graph.edges_hdf.columns
            if col in self._default_edge_column_filter
        ]
        edge_cols.append("EDGE_ORDER")

        return DataFrame(self._results.get("o_edges", None), columns=edge_cols)

    @property
    def weight(self) -> float:
        """
        Weight of the shortest path. Returns 1.0 if no weight column
        was provided to the `execute()` call. Returns -1.0 as initial
        value.

        Returns
        -------
        float
            Weight of the shortest path.
        """
        scalars = self._results.get("o_scalars", None)

        if scalars is None:
            return -1.0
        else:
            return scalars[0][0]
