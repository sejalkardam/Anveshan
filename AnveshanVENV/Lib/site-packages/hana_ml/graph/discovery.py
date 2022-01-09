"""
This module contains functions to discover the graph workspaces of a
given connection to SAP HANA

The following functions are available:

* :func: `discover_graph_workspaces`
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

import pandas as pd
from hdbcli import dbapi

from hana_ml import ConnectionContext
from hana_ml.graph.constants import (
    G_SCHEMA,
    GWS_NAME,
    V_SCHEMA,
    V_TABLE,
    E_SCHEMA,
    E_TABLE,
    EDGE_SOURCE,
    EDGE_TARGET,
    EDGE_KEY,
    VERTEX_KEY,
    CREATE_TIME_STAMP,
    USER_NAME,
    IS_VALID,
)

logger = logging.getLogger(__name__)

CLOUD = "CLOUD"
ONPREM = "ONPREM"


def _cloud_or_on_prem(connection_context: ConnectionContext):
    """
    Internal function to check if the connection context points to a
    Cloud installation or an on premise installation

    :param connection_context:
    :return: DB Version
    """
    v_str = connection_context.hana_version()
    if "CE" in v_str[v_str.find("(") : v_str.find(")")] or int(v_str[0]) > 3:
        return CLOUD
    return ONPREM


def discover_graph_workspaces(connection_context: ConnectionContext):
    """
    Provide a view of the Graph Workspaces (GWS) on a given connection to
    SAP HANA. This provides the basis for creating a HANA graph from
    existing GWS instead of only creating them from vertex and edge tables.
    Use the SYS SQL provided for Graph Workspaces so a user can create a
    HANA graph from one of them. The SQL returns the following per GWS:

        SCHEMA_NAME, WORKSPACE_NAME, CREATE_TIMESTAMP, USER_NAME,
        EDGE_SCHEMA_NAME, EDGE_TABLE_NAME, EDGE_SOURCE_COLUMN_NAME,
        EDGE_TARGET_COLUMN_NAME, EDGE_KEY_COLUMN_NAME, VERTEX_SCHEMA_NAME,
        VERTEX_TABLE_NAME, VERTEX_KEY_COLUMN_NAME, IS_VALID.

    Due to the differences in Cloud and On-Prem Graph workspaces, the SQL
    creation requires different methods to derive the same summary pattern
    for GWS as defined above. For this reason, 2 internal functions return
    the summary.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection to the given SAP HANA Database and implied Graph
        Workspace.

    Returns
    -------
    list
        The list of tuples returned by fetchall but with headers included
        and as a dict.

    """


    DISCOVER_CLOUD = """
                        SELECT GW.*, GWC.* 
                        FROM SYS.GRAPH_WORKSPACES AS GW
                        LEFT OUTER JOIN SYS.GRAPH_WORKSPACE_COLUMNS AS GWC
                            ON GW.SCHEMA_NAME = GWC.SCHEMA_NAME 
                            AND GW.WORKSPACE_NAME = GWC.WORKSPACE_NAME
                        ORDER BY GW.SCHEMA_NAME, GW.WORKSPACE_NAME, 
                            GWC.ENTITY_TYPE, GWC.ENTITY_ROLE_POSITION;
                    """

    DISCOVER_ON_PREM = "SELECT * FROM SYS.GRAPH_WORKSPACES"

    if not connection_context:
        raise Exception(
            "Cannot determine HANA and version information. Connection is not set."
        )

    with connection_context.connection.cursor() as cur:
        try:
            # Check if the current connected version is cloud or on-prem to decide
            # the proper SQL to run
            if _cloud_or_on_prem(connection_context) == CLOUD:
                # Gets G_SCHEMA(0), WORKSPACE(1), ENTIY_TYPE(2), E_SCH(3), E_TBL(4),
                # ROLE(5), COLUMN(6), IS_VALID(7)
                cur.execute(DISCOVER_CLOUD)
                res = cur.fetchall()
                # Storage for aggregated results
                summary = []

                # Start with the first schema and graphworkspace
                if len(res) < 1:
                    raise ValueError("There are no graph workspaces in this schema")
                i_schema = res[0].column_names.index("SCHEMA_NAME")
                i_isvalid = res[0].column_names.index("IS_VALID")
                i_wrkspace = res[0].column_names.index("WORKSPACE_NAME")
                i_ent_type = res[0].column_names.index("ENTITY_TYPE")
                i_ent_role = res[0].column_names.index("ENTITY_ROLE")
                i_sch_name = res[0].column_names.index("ENTITY_SCHEMA_NAME")
                i_tbl_name = res[0].column_names.index("ENTITY_TABLE_NAME")
                i_col_name = res[0].column_names.index("ENTITY_COLUMN_NAME")
                cur_sch_wks = "{}{}".format(res[0][0], res[0][1])
                cur_gws = {
                    G_SCHEMA: res[0][i_schema],
                    GWS_NAME: res[0][i_wrkspace],
                    IS_VALID: res[0][i_isvalid],
                }

                for row in res:
                    # Iterate results to combine the workspace descriptors into
                    # a single row for the summary
                    if "{}{}".format(row[i_schema], row[i_wrkspace]) == cur_sch_wks:
                        if row[i_ent_type] == "VERTEX" and row[i_ent_role] == "KEY":
                            cur_gws[V_SCHEMA] = row[i_sch_name]
                            cur_gws[V_TABLE] = row[i_tbl_name]
                            cur_gws[VERTEX_KEY] = row[i_col_name]
                        elif row[i_ent_type] == "EDGE" and row[i_ent_role] == "KEY":
                            cur_gws[E_SCHEMA] = row[i_sch_name]
                            cur_gws[E_TABLE] = row[i_tbl_name]
                            cur_gws[EDGE_KEY] = row[i_col_name]
                        elif row[i_ent_type] == "EDGE" and row[i_ent_role] == "SOURCE":
                            cur_gws[EDGE_SOURCE] = row[i_col_name]
                        elif row[i_ent_type] == "EDGE" and row[i_ent_role] == "TARGET":
                            cur_gws[EDGE_TARGET] = row[i_col_name]
                    else:
                        summary.append(cur_gws)
                        cur_sch_wks = "{}{}".format(row[i_schema], row[i_wrkspace])
                        cur_gws = {G_SCHEMA: row[i_schema], GWS_NAME: row[i_wrkspace]}
                # Append the last started since the append only occurs if the
                # current is different than the last.
                summary.append(cur_gws)

            else:
                cur.execute(DISCOVER_ON_PREM)
                res = cur.fetchall()
                summary = [
                    {
                        G_SCHEMA: gws[0],
                        GWS_NAME: gws[1],
                        CREATE_TIME_STAMP: gws[2],
                        USER_NAME: gws[3],
                        E_SCHEMA: gws[4],
                        E_TABLE: gws[5],
                        EDGE_SOURCE: gws[6],
                        EDGE_TARGET: gws[7],
                        EDGE_KEY: gws[8],
                        V_SCHEMA: gws[9],
                        V_TABLE: gws[10],
                        VERTEX_KEY: gws[11],
                        IS_VALID: gws[12],
                    }
                    for gws in res
                ]

            return pd.DataFrame(summary)

        except dbapi.Error:
            raise Exception(
                "Unable to get HANA version."
                " please check with your database administrator"
            )

        except dbapi.ProgrammingError:
            logger.error("No result set")


def discover_graph_workspace(
    connection_context: ConnectionContext, workspace_name: str, schema: str = None
):
    """
    Provide detail information about a specific Graph Workspace
    The function returns the following per GWS:

    SCHEMA_NAME, WORKSPACE_NAME, EDGE_SCHEMA_NAME, EDGE_TABLE_NAME,
    EDGE_SOURCE_COLUMN_NAME, EDGE_TARGET_COLUMN_NAME, EDGE_KEY_COLUMN_NAME,
    VERTEX_SCHEMA_NAME, VERTEX_TABLE_NAME, VERTEX_KEY_COLUMN_NAME.

    Parameters
    ----------
    connection_context : ConnectionContext
        Connection to the given SAP HANA Database and implied Graph
        Workspace.

    workspace_name : str
        Workspace name to be discovered

    schema : str
        Schema of the workspace. If none is provided, the schema of the
            `connection_context` is used

    Returns
    -------
    dict
        Dictionary with the workspace attributes

    """
    if not schema:
        schema = connection_context.get_current_schema()

    if _cloud_or_on_prem(connection_context) == CLOUD:
        sql = """
                            SELECT ENTITY_TYPE, ENTITY_SCHEMA_NAME, 
                                   ENTITY_TABLE_NAME, ENTITY_COLUMN_NAME, ENTITY_ROLE
                            FROM SYS.GRAPH_WORKSPACE_COLUMNS
                           WHERE SCHEMA_NAME = '{}'
                             AND WORKSPACE_NAME = '{}'
                           ORDER BY ENTITY_TYPE, ENTITY_ROLE_POSITION
                        """.format(
            schema, workspace_name
        )
    else:
        sql = """
                    SELECT EDGE_TABLE_NAME, EDGE_SCHEMA_NAME, EDGE_KEY_COLUMN_NAME,
                           EDGE_SOURCE_COLUMN_NAME, EDGE_TARGET_COLUMN_NAME,
                           VERTEX_TABLE_NAME, VERTEX_SCHEMA_NAME, VERTEX_KEY_COLUMN_NAME
                      FROM SYS.GRAPH_WORKSPACES
                     WHERE SCHEMA_NAME = '{}'
                       AND WORKSPACE_NAME = '{}'
                """.format(
            schema, workspace_name
        )

    meta_df = connection_context.sql(sql).collect()

    if len(meta_df) == 0:
        raise ValueError(
            "No graph workspace with name '{}' found in schema '{}'".format(
                workspace_name, schema
            )
        )

    meta = {}
    if _cloud_or_on_prem(connection_context) == CLOUD:
        # The metadata schema in cloud is row based in a specific columns table
        for index, row in meta_df.iterrows():  # pylint: disable=unused-variable
            if row["ENTITY_TYPE"] == "EDGE":
                meta["EDGE_TABLE_NAME"] = row["ENTITY_TABLE_NAME"]
                meta["EDGE_SCHEMA_NAME"] = row["ENTITY_SCHEMA_NAME"]

                if row["ENTITY_ROLE"] == "KEY":
                    meta["EDGE_KEY_COLUMN_NAME"] = row["ENTITY_COLUMN_NAME"]
                elif row["ENTITY_ROLE"] == "SOURCE":
                    meta["EDGE_SOURCE_COLUMN_NAME"] = row["ENTITY_COLUMN_NAME"]
                elif row["ENTITY_ROLE"] == "TARGET":
                    meta["EDGE_TARGET_COLUMN"] = row["ENTITY_COLUMN_NAME"]
            elif row["ENTITY_TYPE"] == "VERTEX":
                meta["VERTEX_TABLE_NAME"] = row["ENTITY_TABLE_NAME"]
                meta["VERTEX_SCHEMA_NAME"] = row["ENTITY_SCHEMA_NAME"]

                if row["ENTITY_ROLE"] == "KEY":
                    meta["VERTEX_KEY_COLUMN_NAME"] = row["ENTITY_COLUMN_NAME"]
    else:
        # The on prem version is just a table row
        meta["EDGE_TABLE_NAME"] = meta_df["EDGE_TABLE_NAME"][0]
        meta["EDGE_SCHEMA_NAME"] = meta_df["EDGE_SCHEMA_NAME"][0]
        meta["EDGE_KEY_COLUMN_NAME"] = meta_df["EDGE_KEY_COLUMN_NAME"][0]
        meta["EDGE_SOURCE_COLUMN_NAME"] = meta_df["EDGE_SOURCE_COLUMN_NAME"][0]
        meta["EDGE_TARGET_COLUMN"] = meta_df["EDGE_TARGET_COLUMN_NAME"][0]
        meta["VERTEX_TABLE_NAME"] = meta_df["VERTEX_TABLE_NAME"][0]
        meta["VERTEX_SCHEMA_NAME"] = meta_df["VERTEX_SCHEMA_NAME"][0]
        meta["VERTEX_KEY_COLUMN_NAME"] = meta_df["VERTEX_KEY_COLUMN_NAME"][0]

    return meta
