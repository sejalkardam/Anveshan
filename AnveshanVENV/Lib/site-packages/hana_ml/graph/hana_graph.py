"""
This module represents a database set of HANA Dataframes that are
the edge and vertex tables of a HANA Graph.

The following classes and functions are available:

    * :class:`Graph`
    * :class:`Path`
"""

# pylint: disable=bad-continuation
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-branches
# pylint: disable=invalid-name
# pylint: disable=too-many-lines
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-statements
# pylint: disable=consider-using-f-string
import logging
import textwrap

import hdbcli
import pandas as pd
from hdbcli import dbapi

from .constants import (
    DEFAULT_DIRECTION,
    DIRECTION_INCOMING,
    DIRECTION_OUTGOING,
    DIRECTION_ANY,
)
from .discovery import discover_graph_workspace
from .describer import Describer

from .. import ConnectionContext

logger = logging.getLogger(__name__)


class Graph(object):  # pylint: disable=too-many-public-methods
    """
    Represents a graph consisting of a vertex and edges table that was
    created from a set of pandas dataframes, existing tables that are
    changed into a graph workspace, or through an existing graph workspace.

    At runtime you can access the following attributes:

    * connection_context
    * workspace_schema
    * workspace_name

    * vertex_tbl_schema
    * vertex_tbl_name
    * vertex_key_column
    * vertex_key_col_dtype: DB datatype of the vertex key column
    * vertices_hdf: hana_ml.DataFrame of the vertices

    * edge_tbl_name
    * edge_tbl_schema
    * edge_key_column
    * edge_source_column
    * edge_target_column
    * edge_key_col_dtype: DB datatype of the edge key column
    * edges_hdf: hana_ml.DataFrame of the edges

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    schema : str
        Name of the schema.
    workspace_name : str
        Name that references the HANA Graph workspace.
    """

    def __init__(
            self,
            connection_context: ConnectionContext,
            workspace_name: str,
            schema: str = None,
    ):

        if not schema:
            schema = connection_context.get_current_schema()

        # Get the workspaces in the given connection context and ensure the
        # named space is included.
        meta = discover_graph_workspace(connection_context, workspace_name, schema)

        self.connection_context = connection_context
        self.workspace_schema = schema
        self.workspace_name = workspace_name

        self.vertex_tbl_schema = meta["VERTEX_SCHEMA_NAME"]
        self.vertex_tbl_name = meta["VERTEX_TABLE_NAME"]
        self.vertex_key_column = meta["VERTEX_KEY_COLUMN_NAME"]
        self.vertices_hdf = connection_context.table(
            self.vertex_tbl_name, self.vertex_tbl_schema
        )
        vertex_dt = [
            dtype[1]
            for dtype in self.vertices_hdf.dtypes()
            if dtype[0] == self.vertex_key_column
        ][0]
        if vertex_dt == "NVARCHAR":
            vertex_dt = "NVARCHAR(5000)"
        self.vertex_key_col_dtype = vertex_dt

        self.edge_tbl_name = meta["EDGE_TABLE_NAME"]
        self.edge_tbl_schema = meta["EDGE_SCHEMA_NAME"]
        self.edge_key_column = meta["EDGE_KEY_COLUMN_NAME"]
        self.edge_source_column = meta["EDGE_SOURCE_COLUMN_NAME"]
        self.edge_target_column = meta["EDGE_TARGET_COLUMN"]
        self.edges_hdf = connection_context.table(
            self.edge_tbl_name, self.edge_tbl_schema
        )
        edge_dt = [
            dtype[1]
            for dtype in self.edges_hdf.dtypes()
            if dtype[0] == self.edge_key_column
        ][0]
        if edge_dt == "NVARCHAR":
            edge_dt = "NVARCHAR(5000)"
        self.edge_key_col_dtype = edge_dt

    def __str__(self):
        s = """
            Workspace schema: {}
            Workspace name: {}

            Vertex table schema: {}
            Vertex table name: {}
            Vertex table key column: {}
            Vertex table key column dtype: {}
            Vertex table SQL statement: {}

            Edge table schema: {}
            Edge table name: {}
            Edge table key column: {}
            Edge table key column dtype: {}
            Edge table source columns: {}
            Edge table target column: {}
            Edge table SQL statement: {}
        """.format(
            self.workspace_schema,
            self.workspace_name,
            self.vertex_tbl_schema,
            self.vertex_tbl_name,
            self.vertex_key_column,
            self.vertex_key_col_dtype,
            self.vertices_hdf.select_statement,
            self.edge_tbl_schema,
            self.edge_tbl_name,
            self.edge_key_column,
            self.edge_key_col_dtype,
            self.edge_source_column,
            self.edge_target_column,
            self.edges_hdf.select_statement,
        )

        # Reindent
        return textwrap.dedent(s)

    def __repr__(self):
        return "Graph(connection_context=hana_ml.dataframe.ConnectionContext(), workspace_name='{}', schema='{}')".format(  # pylint: disable=line-too-long
            self.workspace_name, self.workspace_schema
        )

    def describe(self) -> pd.Series:
        """
        Generate descriptive statistics.

        Descriptive statistics include degree, density, counts (edges,
        vertices, self loops, triangles), if it has unconnected nodes...

        The `triangles count` and the `is connected` data are only available in
        the cloud edition. These information will not be available on an
        on-premise installation.

        Returns
        -------
        pandas.Series :
            Statistics
        """
        describer = Describer(graph=self)

        # Count from existing dataframes
        desc = {
            "COUNT(VERTICES)": self.vertices_hdf.count(),
            "COUNT(EDGES)": self.edges_hdf.count(),
        }

        # Merge all to one Series
        desc_series = pd.Series(desc)
        desc_series = desc_series.append(describer.self_loops)
        desc_series = desc_series.append(describer.degree)
        desc_series = desc_series.append(describer.density)

        try:
            # Not supported on on-prem, so it won't be added
            desc_series = desc_series.append(describer.triangles_count)
            desc_series = desc_series.append(describer.is_connected)
        except hdbcli.dbapi.NotSupportedError as err:
            logger.warning(err)
            pass

        return desc_series

    def degree_distribution(self) -> pd.DataFrame:
        """
        Generate the degree distribution of the graph.

        Returns
        -------
        pandas.DataFrame :
            Degree distribution
        """
        degree_sql = """
            WITH degs AS
              (SELECT o.id,
                      coalesce(o.out_deg, 0) AS out_deg,
                      coalesce(i.in_deg, 0) AS in_deg,
                      coalesce(o.out_deg, 0) + coalesce(i.in_deg, 0) AS deg
               FROM
                 (SELECT "{source_col}" AS "ID", COUNT(*) AS OUT_DEG
                  FROM "{schema}"."{edges}"
                  GROUP BY "{source_col}") AS o
               FULL OUTER JOIN
                 (SELECT "{target_col}" AS "ID", COUNT(*) AS IN_DEG
                  FROM "{schema}"."{edges}"
                  GROUP BY "{target_col}") AS i ON o.id = i.id)
            SELECT deg,
                   count(*) AS "COUNT"
            FROM degs
            GROUP BY deg
            ORDER BY deg ASC
        """.format(
            schema=self.edge_tbl_schema,
            edges=self.edge_tbl_name,
            source_col=self.edge_source_column,
            target_col=self.edge_target_column,
        )

        return self.connection_context.sql(degree_sql).collect()

    def drop(self, include_vertices=False, include_edges=False):
        """
        Drops the current graph workspace and all the associated procedures.

        You can also specify to delete the vertices and edges tables if
        required.

        **Note:** The instance of the graph object is not usable anymore
        afterwards.

        Parameters
        ----------
        include_vertices : bool, optional, default: False
            Also drop the Vertices Table

        include_edges : bool, optional, default: False
            Also drop the Edge Table
        """

        def _is_based_on_view(schema: str, name: str) -> bool:
            """Check if a dataframe is based on a view"""
            sql = "SELECT COUNT(*) FROM SYS.VIEWS WHERE SCHEMA_NAME='{}' AND VIEW_NAME='{}'".format(
                schema, name
            )

            with self.connection_context.connection.cursor() as cur:
                cur.execute(sql)
                meta = cur.fetchall()

            return meta[0][0] == 1

        try:
            self.connection_context.connection.cursor().execute(
                """DROP GRAPH WORKSPACE "{}"."{}" """.format(
                    self.workspace_schema, self.workspace_name
                )
            )
        except dbapi.Error as error:
            if "invalid graph workspace name:" in error.errortext:
                pass
            else:
                logger.error(error.errortext)

        if include_edges:
            try:
                if _is_based_on_view(self.edge_tbl_schema, self.edge_tbl_name):
                    sql = 'DROP VIEW "{}"."{}"'.format(
                        self.edge_tbl_schema, self.edge_tbl_name
                    )
                else:
                    sql = 'DROP TABLE "{}"."{}"'.format(
                        self.edge_tbl_schema, self.edge_tbl_name
                    )

                self.connection_context.connection.cursor().execute(sql)
            except dbapi.Error as error:
                logger.warning(error.errortext)

        if include_vertices:
            try:
                if _is_based_on_view(self.vertex_tbl_schema, self.vertex_tbl_name):
                    sql = 'DROP VIEW "{}"."{}"'.format(
                        self.vertex_tbl_schema, self.vertex_tbl_name
                    )
                else:
                    sql = 'DROP TABLE "{}"."{}"'.format(
                        self.vertex_tbl_schema, self.vertex_tbl_name
                    )

                self.connection_context.connection.cursor().execute(sql)
            except dbapi.Error as error:
                logger.warning(error.errortext)

    def has_vertices(self, vertices) -> bool:
        """
        check if they list of vertices are in the graph.

        Edge case is possible where source tables are not up to date of the workspace.

        Parameters
        ----------
        vertices : list
            Vertex keys expected to be in the graph.

        Returns
        -------
        bool :
            True if the vertices exist otherwise False.

        """
        vertex_str = ", ".join(["'{}'".format(vertex) for vertex in vertices])
        cur = self.connection_context.connection.cursor()

        try:
            sql = """
                SELECT "{key}" FROM "{sch}"."{tbl}" where "{tbl}"."{key}" IN ({vertex_str})
            """.format(
                key=self.vertex_key_column,
                sch=self.vertex_tbl_schema,
                tbl=self.vertex_tbl_name,
                vertex_str=vertex_str,
            )
            cur.execute(sql)
            vertex_check = cur.fetchall()
        except dbapi.ProgrammingError as ex:
            logger.error(ex)
            raise

        # Find the missing ones
        if len(vertex_check) < len(vertices):
            missing = ", ".join(
                list(
                    filter(
                        lambda vertex_key: vertex_key
                                           not in [key[0] for key in vertex_check],
                        [str(key) for key in vertices],
                    )
                )
            )

            logger.warning(
                "['%s'] is/are not recognized key(s) in '%s'",
                missing,
                self.vertex_tbl_name,
            )

            return False

        return True

    def vertices(self, vertex_key=None) -> pd.DataFrame:
        """
        Get the table representing vertices within a graph. If there is
        a vertex, check it.

        Parameters
        ----------
        vertex_key : optional
            Vertex key expected to be in the graph.

        Returns
        -------
        pd.Dataframe
            The dataframe is empty, if no vertices are found.

        """
        if not vertex_key:
            pdf = self.vertices_hdf.collect()
        else:
            pdf = self.vertices_hdf.filter(
                "\"{}\" = '{}'".format(self.vertex_key_column, vertex_key)
            ).collect()

        return pdf

    def edges(
            self, vertex_key=None, edge_key=None, direction=DEFAULT_DIRECTION
    ) -> pd.DataFrame:
        """
        Get the table representing edges within a graph. If there is a
        vertex_key, then only get the edges respective to that vertex.

        Parameters
        ----------
        vertex_key : optional
            Vertex key from which to get edges.

            Defaults to None.

        edge_key : optional
            Edge key from which to get edges.

            Defaults to None.

        direction : str, optional
            OUTGOING, INCOMING, or ANY which determines the algorithm
            results. Only applicable if vertex_key is not None.

            Defaults to OUTGOING.

        Returns
        -------
        pd.Dataframe

        """
        pdf = None

        if vertex_key:
            if direction == DIRECTION_ANY:
                where = """"{src}" = '{v_key}' OR "{tgt}" = '{v_key}'""".format(
                    src=self.edge_source_column,
                    tgt=self.edge_target_column,
                    v_key=vertex_key,
                )

                pdf = self.edges_hdf.filter(where).collect()
            elif direction == DIRECTION_INCOMING:
                pdf = self.in_edges(vertex_key=vertex_key)
            elif direction == DIRECTION_OUTGOING:
                pdf = self.out_edges(vertex_key=vertex_key)
        elif edge_key:
            where = """"{key_col}" = '{e_key}' """.format(
                key_col=self.edge_key_column, e_key=edge_key,
            )

            pdf = self.edges_hdf.filter(where)
        elif not vertex_key and not edge_key:
            pdf = self.edges_hdf.collect()

        return pdf

    def in_edges(self, vertex_key) -> pd.DataFrame:
        """
        Get the table representing edges within a graph filtered on a
        vertex_key and its incoming edges.

        Parameters
        ----------
        vertex_key : str
            Vertex key from which to get edges.

        Returns
        -------
        pd.Dataframe

        """
        return self.edges_hdf.filter(
            """"{tgt}" = '{v_key}'""".format(
                tgt=self.edge_target_column, v_key=vertex_key
            )
        ).collect()

    def out_edges(self, vertex_key):
        """
        Get the table representing edges within a graph filtered on a
        vertex_key and its outgoing edges.

        Parameters
        ----------
        vertex_key : str
            Vertex key from which to get edges.

        Returns
        -------
        pd.Dataframe

        """
        return self.edges_hdf.filter(
            """"{src}" = '{v_key}'""".format(
                src=self.edge_source_column, v_key=vertex_key
            )
        ).collect()

    def source(self, edge_key) -> pd.DataFrame:
        """
        Get the vertex that is the source/from/origin/start point of an
        edge.

        Parameters
        ----------
        edge_key :
            Edge key from which to get source vertex.

        Returns
        -------
        pd.Dataframe

        """
        cur = self.connection_context.connection.cursor()

        cur.execute(
            """
                SELECT "{src}" FROM "{sch}"."{edge_tbl}" WHERE "{e_key_col}" = '{e_key}'
            """.format(
                src=self.edge_source_column,
                sch=self.edge_tbl_schema,
                edge_tbl=self.edge_tbl_name,
                e_key_col=self.edge_key_column,
                e_key=edge_key,
            )
        )
        result = cur.fetchone()

        if result:
            return self.vertices(vertex_key=result[0])
        else:
            # Return an empty DF with the correct structure
            return self.vertices_hdf.filter("0=1").collect()

    def target(self, edge_key) -> pd.DataFrame:
        """
        Get the vertex that is the source/from/origin/start point of an
        edge.

        Parameters
        ----------
        edge_key :
            Edge key from which to get source vertex.

        Returns
        -------
        pd.Dataframe

        """
        cur = self.connection_context.connection.cursor()

        cur.execute(
            """
                SELECT "{tgt}" FROM "{sch}"."{edge_tbl}" WHERE "{e_key_col}" = '{e_key}'
            """.format(
                tgt=self.edge_target_column,
                sch=self.edge_tbl_schema,
                edge_tbl=self.edge_tbl_name,
                e_key_col=self.edge_key_column,
                e_key=edge_key,
            )
        )

        result = cur.fetchone()

        if result:
            return self.vertices(vertex_key=result[0])
        else:
            # Return an empty DF with the correct structure
            return self.vertices_hdf.filter("0=1").collect()

    def subgraph(
            self,
            workspace_name,
            schema: str = None,
            vertices_filter: str = None,
            edges_filter: str = None,
            force: bool = False,
    ) -> "Graph":
        """
        Creates a vertices or edges induced subgraph based on SQL filters
        to the respective data frame. The SQL filter has to be valid
        for the dataframe, that will be filtered, otherwise you'll get
        a runtime exception.

        You can provide either a filter to the vertices dataframe or to
        the edges dataframe (not both). Based on the provided filter,
        a new consistent graph workspace is created based on HANA DB views.

        If you for example create an edge filter, a db view for edges based
        on this filter is created. In addation, a db view for the vertices
        is created, which filters the original vertices table, so that it
        only contains the vertecis included in the filtered edges view.

        **Note:** The view names are generated based on
        `<workspace name>_SGE_VIEW` and `<workspace name>_SGV_VIEW`


        Parameters
        ----------
        workspace_name : str
            Name of the workspace expected in the SAP HANA Graph workspaces
            of the ConnectionContext.
        schema : str
            Schema name of the workspace. If this value is not provided or set to
            None, then the value defaults to the ConnectionContext's current schema.

            Defaults to the current schema.
        vertices_filter : str
            SQL filter clause, that will be applied to the vertices dataframe
        edges_filter : str
            SQL filter clause, that will be applied to the edges dataframe
        force : bool, optional
            If force is True, then an existing workspace is overwritten
            during the creation process.

            Defaults to False.

        Returns
        -------
        Graph
            A virtual HANA Graph with functions inherited from the individual
            vertex and edge HANA Dataframes.

        Examples
        --------
        >>> sg = my_graph.subgraph(
        >>>     "sg_geo_filtered",
        >>>     vertices_filter="\"lon_lat_GEO\".ST_Distance(ST_GeomFromWKT( 'POINT(-93.09230195104271 27.810864761841017)', 4326)) < 40000",  # pylint: disable=line-too-long
        >>> )
        >>> print(sg)

        >>> sg = my_graph.subgraph(
        >>>     "sg_test", vertices_filter='"value" BETWEEN 300 AND 400'
        >>> )
        >>> print(sg)

        >>> sg = my_graph.subgraph("sg_test", edges_filter='"rating" > 4')
        >>> print(sg)
        """
        if vertices_filter and edges_filter:
            raise ValueError(
                "Please provide either a vertices filter or an edges filter. Not both."
            )

        if not schema:
            schema = self.workspace_schema

        if vertices_filter:
            # Create Vertices View
            v_df = self.vertices_hdf.filter(vertices_filter).save(
                where=(schema, "{}_SGV_VIEW".format(workspace_name)),
                table_type="VIEW",
                force=True,
            )

            # Create Edges View based on Vertices View
            sql = """"{}" IN (SELECT "{}" FROM "{}"."{}")
                  AND "{}" IN (SELECT "{}" FROM "{}"."{}")
            """.format(
                self.edge_source_column,
                self.vertex_key_column,
                v_df.source_table["SCHEMA_NAME"],
                v_df.source_table["TABLE_NAME"],
                self.edge_target_column,
                self.vertex_key_column,
                v_df.source_table["SCHEMA_NAME"],
                v_df.source_table["TABLE_NAME"],
            )

            e_df = (
                self.edges_hdf.filter(sql)
                    .distinct()
                    .save(
                    where=(schema, "{}_SGE_VIEW".format(workspace_name)),
                    table_type="VIEW",
                    force=True,
                )
            )

        if edges_filter:
            # Create Edges View
            e_df = self.edges_hdf.filter(edges_filter).save(
                where=(schema, "{}_SGE_VIEW".format(workspace_name)),
                table_type="VIEW",
                force=True,
            )

            # Create Vertices View based on Edges View
            sql = """"{}" IN (SELECT "{}" AS VERTEX_ID
                               FROM "{}"."{}"
                              UNION
                             SELECT "{}" AS VERTEX_ID
                               FROM "{}"."{}")""".format(
                self.vertex_key_column,
                self.edge_source_column,
                e_df.source_table["SCHEMA_NAME"],
                e_df.source_table["TABLE_NAME"],
                self.edge_target_column,
                e_df.source_table["SCHEMA_NAME"],
                e_df.source_table["TABLE_NAME"],
            )
            v_df = self.vertices_hdf.filter(sql).save(
                where=(schema, "{}_SGV_VIEW".format(workspace_name)),
                table_type="VIEW",
                force=True,
            )

        # pylint shouldn't return a cyclic import. It seems to be caused
        # but the `Graph` import at the top of the `factory.py` module
        # which should only be evaluated when type hinting. The "real"
        # `Graph` import happens inside the factory method, which shouldn't
        # lead to any problem.
        # pylint: disable=cyclic-import
        from .factory import (  # pylint: disable=import-outside-toplevel
            create_graph_from_hana_dataframes,
        )

        # pylint: enable=cyclic-import

        return create_graph_from_hana_dataframes(
            connection_context=self.connection_context,
            vertices_df=v_df,
            vertex_key_column=self.vertex_key_column,
            edges_df=e_df,
            edge_key_column=self.edge_key_column,
            workspace_name=workspace_name,
            schema=schema,
            edge_source_column=self.edge_source_column,
            edge_target_column=self.edge_target_column,
            force=force,
        )
