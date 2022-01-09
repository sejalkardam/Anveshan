# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from .algorithm_base import AlgorithmBase
from .. import Graph


class WeaklyConnectedComponents(AlgorithmBase):
    """
    Identifies (weakly) connected components.

    An undirected graph is called connected if each of its vertices is
    reachable of every other ones. A directed graph is called weakly
    connected if the undirected graph, naturally derived from that, is
    connected.
    Being weakly connected is an equivalence relation between vertices
    and therefore the weakly connected components (wcc) of the graph form
    a partition on the vertex set.


    The induced subgraphs on these subsets are the weakly connected
    components.
    Note, that each vertex end each edge of the graph is part of exactly one wcc.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> cc = hga.WeaklyConnectedComponents(graph=g).execute()
    >>>
    >>> print("Vertices", cc.vertices)
    >>> print("Components", cc.components)
    >>> print("Number of Components", cc.components_count)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
            DO (
                OUT o_vertices TABLE ({vertex_columns}, "COMPONENT" BIGINT) => ?,
                OUT o_hist TABLE ("COMPONENT" BIGINT, "NUMBER_OF_VERTICES" BIGINT) => ?,
                OUT o_scalars TABLE ("NUMBER_OF_COMPONENTS" BIGINT) => ?
            )
            LANGUAGE GRAPH
            BEGIN
                GRAPH G = Graph("{schema}", "{workspace}");
                ALTER g ADD TEMPORARY VERTEX ATTRIBUTE (BIGINT "COMPONENT" = 0L);
                BIGINT i = 0L;
                FOREACH v_start IN Vertices(:g){{
                    IF(:v_start."COMPONENT" == 0L) {{
                        i = :i + 1L;
                        MULTISET<Vertex> m_reachable_vertices = REACHABLE_VERTICES(:g, :v_start, 'ANY');
                        o_hist."COMPONENT"[:i] = :i;
                        o_hist."NUMBER_OF_VERTICES"[:i] = COUNT(:m_reachable_vertices);
                        FOREACH v_reachable IN :m_reachable_vertices {{
                            v_reachable."COMPONENT" = :i;
                        }}
                    }}
                }}
                o_vertices = SELECT {vertex_select}, :v."COMPONENT" FOREACH v IN Vertices(:g);
                o_scalars."NUMBER_OF_COMPONENTS"[1L] = :i;
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

    def execute(self) -> "WeaklyConnectedComponents":  # pylint: disable=arguments-differ, useless-super-delegation
        """
        Executes the connected component.

        Returns
        -------
        WeaklyConnectedComponents
            WeaklyConnectedComponents object instance
        """
        return super(WeaklyConnectedComponents, self).execute()

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains the [wie in strongly connected components]
        """
        vertex_cols = [
            col
            for col in self._graph.vertices_hdf.columns
            if col in self._default_vertices_column_filter
        ]
        vertex_cols.append("COMPONENT")

        return DataFrame(self._results.get("o_vertices", None), columns=vertex_cols)

    @property
    def components(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains connected components and
            number of vertices in each component.
        """
        vertex_cols = ["COMPONENT", "NUMBER_OF_VERTICES"]

        return DataFrame(self._results.get("o_hist", None), columns=vertex_cols)

    @property
    def components_count(self) -> int:
        """
        Returns
        -------
        int
            The number of weakly connected components in the graph.
        """
        scalars = self._results.get("o_scalars", None)

        if scalars is None:
            return 0
        else:
            return scalars[0][0]
