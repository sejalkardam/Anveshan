""" Module that contains the implementation for Graph.describe"""
import pandas as pd
# pylint: disable=consider-using-f-string

class Describer:
    """
    Internal class extracting the functions for getting the describe
    statistic for a graph
    """

    def __init__(self, graph):
        self._graph = graph

    @property
    def self_loops(self) -> pd.Series:
        """ Self Loops in the graph"""
        selfloops_sql = """
            SELECT count(*) AS "COUNT(SELF_LOOPS)"
            FROM "{schema}"."{edges}"
            WHERE "{source_col}" = "{target_col}"
        """.format(
            schema=self._graph.edge_tbl_schema,
            edges=self._graph.edge_tbl_name,
            source_col=self._graph.edge_source_column,
            target_col=self._graph.edge_target_column,
        )
        selfloops_df = self._graph.connection_context.sql(selfloops_sql).collect()

        return selfloops_df.iloc[0]

    @property
    def density(self) -> pd.Series:
        """ Densitiy of the Grpah """
        density_sql = """
            SELECT num_edges/(num_vertices*(num_vertices-1)) AS density
            FROM
              (SELECT count(*) AS num_edges
               FROM "{e_schema}"."{edges}") AS e,
              (SELECT count(*) AS num_vertices
               FROM "{v_schema}"."{vertices}") AS v
        """.format(
            e_schema=self._graph.edge_tbl_schema,
            edges=self._graph.edge_tbl_name,
            v_schema=self._graph.vertex_tbl_schema,
            vertices=self._graph.vertex_tbl_name,
        )
        density_df = self._graph.connection_context.sql(density_sql).collect()
        return density_df.iloc[0]

    @property
    def degree(self) -> pd.Series:
        """ Degree of the graph"""
        degree_sql = """
            SELECT min(out_deg), min(in_deg), min(deg), max(out_deg),
                   max(in_deg), max(deg), avg(out_deg), avg(in_deg), avg(deg)
            FROM
              (SELECT o.id, COALESCE(o.out_deg, 0) AS out_deg,
                      COALESCE(i.in_deg, 0) AS in_deg,
                      COALESCE(o.out_deg, 0) + COALESCE(i.in_deg, 0) AS deg
               FROM
                 (SELECT "{source_col}" AS "ID", COUNT(*) AS OUT_DEG
                  FROM "{schema}"."{edges}"
                  GROUP BY "{source_col}") AS o
               FULL OUTER JOIN
                 (SELECT "{target_col}" AS "ID", COUNT(*) AS IN_DEG
                  FROM "{schema}"."{edges}"
                  GROUP BY "{target_col}") AS i ON o.id = i.id)
        """.format(
            source_col=self._graph.edge_source_column,
            target_col=self._graph.edge_target_column,
            schema=self._graph.edge_tbl_schema,
            edges=self._graph.edge_tbl_name,
        )
        degree_df = self._graph.connection_context.sql(degree_sql).collect()

        return degree_df.iloc[0]

    @property
    def triangles_count(self):
        """ Count the triangles in the graph"""
        sql = """
            DO(
                OUT o_scalars TABLE ("TRIANGLES_COUNT" BIGINT) => ?
            )
            LANGUAGE GRAPH
            BEGIN
                GRAPH g = Graph("{schema}", "{workspace}");
                MULTISET<Vertex> m_n = Multiset<Vertex>(:g);
                BIGINT triangleCount = 0L;
                FOREACH v IN Vertices(:g){{
                    m_n = Neighbors(:g, :v, 1, 1, 'ANY');
                    triangleCount = :triangleCount + COUNT(EDGES(:g, :m_n, :m_n));
                }}
                o_scalars."TRIANGLES_COUNT"[1L] = :triangleCount / 3L;
            END;
        """.format(
            schema=self._graph.workspace_schema, workspace=self._graph.workspace_name
        )

        cur = self._graph.connection_context.connection.cursor()
        cur.executemany(sql)

        return pd.Series({"COUNT(TRIANGLES)": cur.fetchall()[0][0]})

    @property
    def is_connected(self):
        """ Are there any unconnected vertices in the graph"""
        sql = """
            DO (
                OUT o_scalars TABLE ("IS_CONNECTED" INT) => ?
            ) 
            LANGUAGE GRAPH
            BEGIN
                GRAPH g = Graph("{schema}", "{workspace}");
                BIGINT number_of_nodes = COUNT(VERTICES(:g));
                IF (:number_of_nodes == 0L) {{ return; }}
                SEQUENCE<Vertex> s_v = Sequence<Vertex>(Vertices(:g));
                INT isConnected = 0;
                IF (COUNT(REACHABLE_VERTICES(:g, :s_v[1L], 'ANY')) == :number_of_nodes) {{
                    isConnected = 1;
                }}
                o_scalars."IS_CONNECTED"[1L] = :isConnected;
            END;
        """.format(
            schema=self._graph.workspace_schema, workspace=self._graph.workspace_name
        )

        cur = self._graph.connection_context.connection.cursor()
        cur.executemany(sql)

        is_connected = cur.fetchall()[0][0]

        return pd.Series({"IS_CONNECTED": is_connected == 1})
