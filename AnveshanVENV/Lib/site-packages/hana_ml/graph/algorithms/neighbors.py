# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from .algorithm_base import AlgorithmBase
from .. import Graph
from ..constants import DEFAULT_DIRECTION, DIRECTIONS


class _NeighborsBase(AlgorithmBase):
    """
    Generic class to read the neighbors with or without the edges

    :param direction:
    :param lower_bound:
    :param start_vertex:
    :param upper_bound:
    :param include_edges:
    :return: Cursor from the SQL query result
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
                    DO (
                        IN i_startVertex {vertex_dtype} => '{start_vertex}', 
                        IN lower_bound BIGINT => {lower_bound}, 
                        IN upper_bound BIGINT => {upper_bound}, 
                        IN i_dir VARCHAR(10) => '{direction}', 
                        OUT o_vertices TABLE ({vertex_columns}) => ?
                        {edges_interface_sql}
                    )
                    LANGUAGE GRAPH
                    BEGIN 
                        GRAPH g = Graph("{schema}", "{workspace}"); 
                        VERTEX v_start = Vertex(:g, :i_startVertex); 
     
                        MULTISET<Vertex> m_neighbors = Neighbors(:g, :v_start, 
                            :lower_bound, :upper_bound, :i_dir);
                        o_vertices = SELECT {vertex_select} FOREACH v IN :m_neighbors;
                        
                        {edges_sql}
                    END;
                  """

        self._graph_script_vars = {
            "start_vertex": ("start_vertex", True, None, None),
            "lower_bound": ("lower_bound", False, int, 1),
            "upper_bound": ("upper_bound", False, int, 1),
            "direction": ("direction", False, str, DEFAULT_DIRECTION),
            "include_edges": ("include_edges", False, bool, False),
            "schema": (None, False, str, self._graph.workspace_schema),
            "workspace": (None, False, str, self._graph.workspace_name),
            "vertex_dtype": (None, False, str, self._graph.vertex_key_col_dtype),
            "vertex_columns": (None, False, str, self._default_vertex_cols()),
            "vertex_select": (None, False, str, self._default_vertex_select("v")),
        }

    def _validate_parameters(self):
        # Version check
        if int(self._graph.connection_context.hana_major_version()) < 4:
            raise EnvironmentError(
                "SAP HANA version is not compatible with this method"
            )

        if self._templ_vals["upper_bound"] and self._templ_vals["lower_bound"]:
            if self._templ_vals["upper_bound"] < self._templ_vals["lower_bound"]:
                raise ValueError(
                    "Max depth (upper_bound) {} is less than min depth (lower_bound) {}".format(
                        self._templ_vals["upper_bound"], self._templ_vals["lower_bound"]
                    )
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

    def _process_parameters(self, arguments):
        super()._process_parameters(arguments)

        # Add edges to sql if requested
        self._templ_vals["edges_interface_sql"] = ""
        self._templ_vals["edges_sql"] = ""

        if self._templ_vals["include_edges"]:
            # Add OUT parameter to the Graph Script
            self._templ_vals[
                "edges_interface_sql"
            ] = ",OUT o_edges TABLE ({edge_columns}) => ?".format(
                edge_columns=self._default_edge_cols()
            )

            # Add Edges SQL Statement to Graph Script
            self._templ_vals[
                "edges_sql"
            ] = """
                            MULTISET<Edge> m_edges = Edges(:g, :m_neighbors, :m_neighbors);
                            o_edges = SELECT {edge_select} FOREACH e IN :m_edges; 
            """.format(
                edge_select=self._default_edge_select("e")
            )


class Neighbors(_NeighborsBase):
    """
    Get a virtual subset of the graph based on a start_vertex and all
    vertices within a lower_bound->upper_bound count of degrees of
    separation.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> nb = hga.Neighbors(graph=g).execute(start_vertex="1")
    >>>
    >>> print("Vertices", nb.vertices)
    """

    def execute(  # pylint: disable=arguments-differ
            self,
            start_vertex: str,
            direction: str = DEFAULT_DIRECTION,
            lower_bound: int = 1,
            upper_bound: int = 1,
    ) -> "Neighbors":
        """
        Executes the calculation of the neighbors.

        Parameters
        ----------
        start_vertex : str
            Source from which the subset is based.

        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm
            results.

            Defaults to OUTGOING.
        lower_bound : int, optional
            The count of degrees of separation from which to start
            considering neighbors. If you want to include the start node
            into consideration, set `lower_bound=0`.

            Defaults to 1.
        upper_bound : int, optional
            The count of degrees of separation at which to end
            considering neighbors.

            Defaults to 1.

        Returns
        -------
        Neighbors
            Neighbors object instance
        """
        return super(Neighbors, self).execute(
            start_vertex=start_vertex,
            direction=direction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_edges=False,
        )

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the vertices
        """
        vertex_cols = [
            col
            for col in self._graph.vertices_hdf.columns
            if col == self._graph.vertex_key_column
        ]

        return DataFrame(self._results.get("o_vertices", None), columns=vertex_cols)


class NeighborsSubgraph(_NeighborsBase):
    """
    Get a virtual subset of the graph based on a start_vertex and all
    vertices within a lower_bound->upper_bound count of degrees of
    separation. The result is similar to :func:`~hana_ml.graph.Graph.neighbors`
    but includes edges which could be useful for visualization.

    **Note:** The edges table also contains edges between neighbors,
    if there are any (not only edges from the start vertex).

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> nb = hga.NeighborsSubgraph(graph=g).execute(start_vertex="1")
    >>>
    >>> print("Vertices", nb.vertices)
    >>> print("Edges", nb.edges)
    """

    def execute(  # pylint: disable=arguments-differ
            self,
            start_vertex: str,
            direction: str = DEFAULT_DIRECTION,
            lower_bound: int = 1,
            upper_bound: int = 1,
    ) -> "NeighborsSubgraph":
        """
        Executes the calculation of the neighbors with edges.

        Parameters
        ----------
        start_vertex : str
            Source from which the subset is based.

        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm
            results.

            Defaults to OUTGOING.
        lower_bound : int, optional
            The count of degrees of separation from which to start
            considering neighbors. If you want to include the start node
            into consideration, set `lower_bound=0`.

            Defaults to 1.
        upper_bound : int, optional
            The count of degrees of separation at which to end
            considering neighbors.

            Defaults to 1.

        Returns
        -------
        NeighborsSubgraph
            NeighborsSubgraph object instance
        """
        return super(NeighborsSubgraph, self).execute(
            start_vertex=start_vertex,
            direction=direction,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            include_edges=True,
        )

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the vertices
        """
        vertex_cols = [
            col
            for col in self._graph.vertices_hdf.columns
            if col in self._default_vertices_column_filter
        ]

        return DataFrame(self._results.get("o_vertices", None), columns=vertex_cols)

    @property
    def edges(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the edges between neighbors
        """
        edge_cols = [
            col
            for col in self._graph.edges_hdf.columns
            if col in self._default_edge_column_filter
        ]

        return DataFrame(self._results.get("o_edges", None), columns=edge_cols)
