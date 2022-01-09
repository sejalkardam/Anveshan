# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from .algorithm_base import AlgorithmBase
from .. import Graph


class TopologicalSort(AlgorithmBase):
    """
    Calculates the topological sort if possible.

    A topological ordering of a directed graph is a linear ordering such
    that for each edge the source vertex comes before the target vertex in
    the row.

    The topological order is not necessarily unique. A directed graph is
    topological sortable if and only if it does not contain any directed
    cycles.

    There are some common used algorithms for finding a topological order
    in the input directed graph. Our implementation is based on the depth-first search.

    In case the directed graph contains a directed cycle, the Boolean
    property :func:`is_sortable` returns with the value 'False'. Otherwise,
    the algorithm returns a topolocical order.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> ts = hga.TopologicalSort(graph=g).execute()
    >>>
    >>> print("Vertices", ts.vertices)
    >>> print("Sortable", ts.is_sortable)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
            DO (
                OUT o_vertices TABLE ({vertex_columns}, "EXIT_ORDER" BIGINT, "DEPTH" BIGINT) => ?,
                OUT o_scalars TABLE ("SORTABLE" INT) => ?
            )
            LANGUAGE GRAPH
            BEGIN
                GRAPH G = Graph("{schema}", "{workspace}");
                ALTER G ADD TEMPORARY VERTEX ATTRIBUTE (BIGINT "IN_DEGREE");
                ALTER G ADD TEMPORARY VERTEX ATTRIBUTE (BIGINT "VISIT_ORDER");
                ALTER G ADD TEMPORARY VERTEX ATTRIBUTE (BIGINT "EXIT_ORDER");
                ALTER G ADD TEMPORARY VERTEX ATTRIBUTE (BIGINT "DEPTH");
                INT v_sortable = 1;
                BIGINT c_visit = 0L;
                BIGINT c_exit = 0L;
                FOREACH v IN VERTICES(:G) {{
                    v."IN_DEGREE" = IN_DEGREE(:v);
                }}
                MULTISET<VERTEX> M_NODES = v IN VERTICES(:G) WHERE :v."IN_DEGREE" == 0L;
                IF (COUNT(:M_NODES) == 0L) {{ 
                    o_scalars."SORTABLE"[1L] = 0;
                    RETURN; 
                }}
                FOREACH v_start in :M_NODES {{
                    TRAVERSE DFS('OUTGOING') :G FROM :v_start
                        ON VISIT VERTEX (VERTEX v_visited, BIGINT dpth) {{
                            IF (:v_visited."VISIT_ORDER" IS NULL) {{
                                c_visit = :c_visit + 1L;
                                v_visited."VISIT_ORDER" = :c_visit;
                                v_visited."DEPTH" = :dpth;
                            }}
                            ELSE {{ END TRAVERSE; }}
                        }}
                        ON EXIT VERTEX (VERTEX v_exited) {{
                            IF (:v_exited."EXIT_ORDER" IS NULL) {{
                                c_exit = :c_exit + 1L;
                                v_exited."EXIT_ORDER" = :c_exit;
                            }}
                        }}
                        ON VISIT EDGE (EDGE e_visited) {{
                            VERTEX S = SOURCE(:e_visited);
                            VERTEX T = TARGET(:e_visited);
                            IF (:T."VISIT_ORDER" IS NOT NULL AND :T."EXIT_ORDER" IS NULL) {{
                                v_sortable = 0;
                                END TRAVERSE ALL;
                            }}
                        }};
                }}
                o_scalars."SORTABLE"[1L] = :v_sortable;
                IF ( :v_sortable == 1 ) {{
                    SEQUENCE<VERTEX> ORDERED_VERTICES = SEQUENCE<VERTEX>(VERTICES(:G)) ORDER BY "EXIT_ORDER" DESC;
                    o_vertices = SELECT {vertex_select}, :v."EXIT_ORDER", :v."DEPTH" FOREACH v IN :ORDERED_VERTICES;
                }}
            END;
        """

        self._graph_script_vars = {
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

    def execute(self) -> "TopologicalSort":  # pylint: disable=arguments-differ, useless-super-delegation
        """
        Executes the topological sort.

        Returns
        -------
        TopologicalSort
            TopologicalSort object instance
        """
        return super(TopologicalSort, self).execute()

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the topologically sorted vertices
        """
        vertex_cols = [
            col
            for col in self._graph.vertices_hdf.columns
            if col in self._default_vertices_column_filter
        ]
        vertex_cols.append("EXIT_ORDER")
        vertex_cols.append("DEPTH")

        return DataFrame(self._results.get("o_vertices", None), columns=vertex_cols)

    @property
    def is_sortable(self) -> bool:
        """
        Flag if the graph is topologically sortable or not. (e.g. false for cyclic graphs)

        Returns
        -------
        bool
            Weight of the shortest path.
        """
        scalars = self._results.get("o_scalars", None)

        if scalars is None:
            return False
        else:
            return scalars[0][0] == 1
