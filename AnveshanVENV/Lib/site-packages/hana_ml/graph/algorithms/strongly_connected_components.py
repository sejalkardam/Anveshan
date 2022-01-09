# pylint: disable=missing-module-docstring
# pylint: disable=consider-using-f-string
from pandas import DataFrame

from .algorithm_base import AlgorithmBase
from .. import Graph


class StronglyConnectedComponents(AlgorithmBase):
    """
    Identifies the strongly connected components of a graph.

    A directed graph is called strongly connected if each of its vertices
    is reachable of every other ones.
    Being strongly connected is an equivalence relation and therefore
    the strongly connected components (scc) of the graph form a partition
    on the vertex set.

    The induced subgraphs on these subsets are the strongly connected
    components. Note, that each vertex of the graph is part of exactly
    one scc, but not every edge is part of any scc (if yes, then in only
    one scc).

    In case each scc contains only one vertex, the graph is a directed
    acyclic graph, as in all strongly connected graphs there should
    exist a cycle on all of its vertices.

    The calculation is started by calling :func:`~execute`.

    Examples
    --------
    >>> import hana_ml.graph.algorithms as hga
    >>> scc = hga.StronglyConnectedComponents(graph=g).execute()
    >>>
    >>> print("Vertices", scc.vertices)
    >>> print("Components", scc.components)
    >>> print("Number of Components", scc.components_count)
    """

    def __init__(self, graph: Graph):
        super().__init__(graph)

        self._graph_script = """
            DO (
                OUT o_vertices TABLE ({vertex_columns}, "COMPONENT" BIGINT) => ?,
                OUT o_sccHistogramm TABLE ("COMPONENT" BIGINT, "NUMBER_OF_VERTICES" BIGINT) => ?,
                OUT o_scalars TABLE ("NUMBER_OF_COMPONENTS" BIGINT) => ?
            )
            LANGUAGE GRAPH
            BEGIN
                GRAPH G = Graph("{schema}", "{workspace}");
                ALTER g ADD TEMPORARY VERTEX ATTRIBUTE(BIGINT "COMPONENT");
                BIGINT componentCounter = 0L;
                Sequence<Sequence<Vertex>> m_scc = STRONGLY_CONNECTED_COMPONENTS(:g);
                FOREACH m_component IN :m_scc {{
                    componentCounter = :componentCounter + 1L;
                    o_sccHistogramm."COMPONENT"[:componentCounter] = :componentCounter;
                    o_sccHistogramm."NUMBER_OF_VERTICES"[:componentCounter] = COUNT(:m_component);
                    FOREACH v IN :m_component {{
                        v."COMPONENT" = :componentCounter;
                    }}
                }}
                o_vertices = SELECT {vertex_select}, :v."COMPONENT" FOREACH v in Vertices(:g);
                o_scalars."NUMBER_OF_COMPONENTS"[1L] = COUNT(:m_scc);
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

    def execute(self) -> "StronglyConnectedComponents":  # pylint: disable=arguments-differ, useless-super-delegation
        """
        Executes Strongly Connected Components.

        Returns
        -------
        StronglyConnectedComponents
            StronglyConnectedComponents object instance
        """
        return super(StronglyConnectedComponents, self).execute()

    @property
    def vertices(self) -> DataFrame:
        """
        Returns
        -------
        pandas.Dataframe
            A Pandas `DataFrame` that contains an assignment of each
            vertex to a strongly connected component.
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
            A Pandas `DataFrame` that contains strongly connected
            components and number of vertices in each component.
        """
        vertex_cols = [
            "COMPONENT",
            "NUMBER_OF_VERTICES"
        ]

        return DataFrame(self._results.get("o_sccHistogramm", None), columns=vertex_cols)

    @property
    def components_count(self) -> int:
        """
        Returns
        -------
        Int
            The number of strongly connected components in the graph.
        """
        scalars = self._results.get("o_scalars", None)

        if scalars is None:
            return 0
        else:
            return scalars[0][0]
