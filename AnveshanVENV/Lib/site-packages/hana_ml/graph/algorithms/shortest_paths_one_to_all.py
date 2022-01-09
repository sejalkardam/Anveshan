# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from .algorithm_base import AlgorithmBase
from .. import Graph
from ..constants import DEFAULT_DIRECTION, DIRECTIONS


class ShortestPathsOneToAll(AlgorithmBase):
    """
    Calculates the shortest paths from a start vertex to all other vertices in the graph.

    The procedure may fail for HANA versions prior to SP05 therefore this
    is checked at execution time.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> spoa = hga.ShortestPathsOneToAll(graph=g).execute(
    >>>     source=2257, direction='OUTGOING', weight='DIST_KM'
    >>> )
    >>>
    >>> print("Vertices", spoa.vertices)
    >>> print("Edges", spoa.edges)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
            DO (
                IN i_startVertex {vertex_dtype} => '{start_vertex}',
                IN i_direction NVARCHAR(10) => '{direction}',
                OUT o_vertices TABLE({vertex_columns}, "DISTANCE" {distance_dtype}) => ?,
                OUT o_edges TABLE({edge_columns}) => ?
                )
            LANGUAGE GRAPH 
            BEGIN
                GRAPH G = Graph("{schema}", "{workspace}");
                VERTEX v_start = Vertex(:g, :i_startVertex);
                
                {weighted_definition}
                
                o_vertices = SELECT {vertex_select}, :v."DISTANCE" FOREACH v IN Vertices(:g_spoa);
                o_edges = SELECT {edge_select} FOREACH e IN Edges(:g_spoa);
            END;
        """

        self._graph_script_vars = {
            "start_vertex": ("source", True, None, None),
            "weight": ("weight", False, str, None),
            "distance_dtype": ("distance_dtype", False, str, None),
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
            self._templ_vals["distance_dtype"] = """DOUBLE"""
            self._templ_vals[
                "weighted_definition"
            ] = """
                    GRAPH g_spoa = SHORTEST_PATHS_ONE_TO_ALL(:g, :v_start, "DISTANCE", (Edge e) => DOUBLE{{ return DOUBLE(:e."{weight}"); }}, :i_direction);
                """.format(
                    weight=self._templ_vals["weight"]
                )
        else:
            self._templ_vals["distance_dtype"] = """BIGINT"""
            self._templ_vals[
                "weighted_definition"
            ] = """
                    GRAPH g_spoa = SHORTEST_PATHS_ONE_TO_ALL(:g, :v_start, "DISTANCE", :i_direction);
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

    def execute(  # pylint: disable=arguments-differ
            self, source: str, weight: str = None, direction: str = DEFAULT_DIRECTION,
    ) -> "ShortestPathsOneToAll":
        """
        Executes the calculation of the shortest paths one to all.

        Parameters
        ----------
        source : str
            Vertex key from which the shortest paths one to all will start.
        weight : str, optional
            Variable for column name to which to apply the weight.

            Defaults to None.
        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm results.

            Defaults to OUTGOING.

        Returns
        -------
        ShortestPathsOneToAll
            ShortestPathOneToAll object instance
        """
        return super(ShortestPathsOneToAll, self).execute(
            source=source, weight=weight, direction=direction
        )

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the vertices and the distance to
            the start vertex
        """
        vertex_cols = [
            col
            for col in self._graph.vertices_hdf.columns
            if col in self._default_vertices_column_filter
        ]
        vertex_cols.append("DISTANCE")

        return DataFrame(self._results.get("o_vertices", None), columns=vertex_cols)

    @property
    def edges(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the edges which are on one
            of the shortest paths
        """
        edge_cols = [
            col
            for col in self._graph.edges_hdf.columns
            if col in self._default_edge_column_filter
        ]

        return DataFrame(self._results.get("o_edges", None), columns=edge_cols)
