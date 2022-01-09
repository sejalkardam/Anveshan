# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from hana_ml.graph import Graph
from hana_ml.graph.algorithms.algorithm_base import AlgorithmBase


class KShortestPaths(AlgorithmBase):
    """
    Given a source and target vertex_key with optional weight, get the
    the `Top-k` shortest paths between them.

    The procedure may fail for HANA versions prior to SP05 therefore this
    is checked at execution time.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> topk = hga.KShortestPaths(graph=g).execute(source="1", target="3", k=3)
    >>>
    >>> print("Paths", topk.paths)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
            DO (
                IN i_startVertex {vertex_dtype} => '{start_vertex}',
                IN i_endVertex {vertex_dtype} => '{end_vertex}',
                IN i_k INT => {k},
                OUT o_paths TABLE ("PATH_ID" INT, "PATH_LENGTH" BIGINT, 
                    "PATH_WEIGHT" DOUBLE, "EDGE_ID" {edge_dtype}, "EDGE_ORDER" INT) => ?
            )
            LANGUAGE GRAPH
            BEGIN
                GRAPH g = Graph("{schema}", "{workspace}");
                VERTEX v_start = Vertex(:g, :i_startVertex);
                VERTEX v_end = Vertex(:g, :i_endVertex);
                
                {weighted_definition}
                
                BIGINT currentResultRow = 1L;
                FOREACH result_path IN (:s_paths) WITH ORDINALITY AS path_id {{
                    FOREACH path_edge in EDGES(:result_path) WITH ORDINALITY AS edge_order {{
                        o_paths."PATH_ID"[:currentResultRow] = INTEGER(:path_id); 
                        o_paths."PATH_LENGTH"[:currentResultRow] = Length(:result_path); 
                        o_paths."PATH_WEIGHT"[:currentResultRow] = DOUBLE(Weight(:result_path)); 
                        o_paths."EDGE_ID"[:currentResultRow] = :path_edge."{edge_key_col}"; 
                        o_paths."EDGE_ORDER"[:currentResultRow] = INTEGER(:edge_order); 
                        currentResultRow = :currentResultRow + 1L; }}
                }}
            END;
        """

        self._graph_script_vars = {
            "start_vertex": ("source", True, None, None),
            "end_vertex": ("target", True, None, None),
            "k": ("k", True, int, None),
            "weight": ("weight", False, str, None),
            "schema": (None, False, str, self._graph.workspace_schema),
            "workspace": (None, False, str, self._graph.workspace_name),
            "vertex_dtype": (None, False, str, self._graph.vertex_key_col_dtype),
            "edge_dtype": (None, False, str, self._graph.edge_key_col_dtype),
            "edge_key_col": (None, False, str, self._graph.edge_key_column),
        }

    def _process_parameters(self, arguments):
        super()._process_parameters(arguments)

        # Construct the WeightedPath part of the SQL Statement depending
        # on if a weight parameter was provided or not
        if self._templ_vals["weight"]:
            self._templ_vals[
                "weighted_definition"
            ] = """
                    SEQUENCE<WeightedPath<DOUBLE>> s_paths = K_Shortest_Paths(:g, 
                        :v_start, :v_end, :i_k, (Edge e) => DOUBLE
                        {{ return :e."{weight}"; }}
                    ); 
                """.format(
                    weight=self._templ_vals["weight"],
                )
        else:
            self._templ_vals[
                "weighted_definition"
            ] = """
                    SEQUENCE<WeightedPath<BIGINT>> s_paths = K_Shortest_Paths(:g,
                        :v_start, :v_end, :i_k);
                """

    def _validate_parameters(self):
        # Version check
        if int(self._graph.connection_context.hana_major_version()) < 4:
            raise EnvironmentError(
                "SAP HANA version is not compatible with this method"
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

    def execute(self, source: str, target: str, k: int, weight: str = None) -> "KShortestPaths":  # pylint: disable=arguments-differ
        """
        Executes the calculation of the top-k shortest paths.

        Parameters
        ----------
        source : str
            Vertex key from which the shortest path will start.
        target : str
            Vertex key from which the shortest path will end.
        k : int
            Number of paths that will be calculated
        weight : str, optional
            Variable for column name to which to apply the weight.

            Defaults to None.

        Returns
        -------
        KShortestPaths
            KShortestPaths object instance
        """
        return super(KShortestPaths, self).execute(
            source=source, target=target, k=k, weight=weight
        )

    @property
    def paths(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the paths
        """
        paths_cols = ["PATH_ID", "PATH_LENGTH", "PATH_WEIGHT", "EDGE_ID", "EDGE_ORDER"]

        return DataFrame(self._results.get("o_paths", None), columns=paths_cols)
