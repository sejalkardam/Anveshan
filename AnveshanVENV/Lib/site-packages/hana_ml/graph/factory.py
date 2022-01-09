"""
This module contains factory functions to create Graph objects based
on different inputs.

The following functions are available:

* :func: `create_graph_from_dataframes`
* :func: `create_hana_graph_from_existing_workspace`
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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from hdbcli import dbapi

from hana_ml import DataFrame
from hana_ml.dataframe import create_dataframe_from_pandas, ConnectionContext

from .constants import EDGE_ID

# Import Graph at this point only for type checking, since it's added
# as type hint return type to the factory methods. At runtime, this
# always is `False`, so it shouldn't lead to cyclic imports
if TYPE_CHECKING:
    from .hana_graph import Graph


logger = logging.getLogger(__name__)


def create_graph_from_edges_dataframe(
    connection_context: ConnectionContext,
    edges_df,
    workspace_name: str,
    schema: str = None,
    edge_source_column: str = "from",
    edge_target_column: str = "to",
    edge_key_column: str = None,
    object_type_as_bin: bool = False,
    drop_exist_tab: bool = True,
    allow_bigint: bool = False,
    force_tables: bool = True,
    force_workspace: bool = True,
    replace: bool = False,
    geo_cols: list = None,  # Spatial variable
    srid: int = 4326,  # Spatial variable
) -> "Graph":
    """
    Create a HANA Graph workspace based on an edge dataframe. The respective
    vertices table is created implicitely based on the `from` and `to` columns
    of the edges.

    Expects either a hana dataframe or pandas dataframe as input for the
    edges table. If it is pandas then it will be transformed into a hana_ml.DataFrame.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    edges_df : pandas.DataFrame or hana_ml.DataFrame
        Table of data containing edges that link keys within the vertex frame.
    workspace_name : str
        Name of the workspace expected in the SAP HANA Graph workspaces
        of the ConnectionContext.
    schema : str
        Schema name of the workspace. If this value is not provided or set to
        None, then the value defaults to the ConnectionContext's current schema.

        Defaults to the current schema.
    edge_source_column : str
        Column name in the e_frame containing only source vertex keys that
        exist within the vertex_key_column of the n_frame.

        Defaults to 'from'.
    edge_target_column : str
        Column name in the e_frame containing the unique id of the edge.

        Defaults to 'to'.
    edge_key_column : str
        Column name in the n_frame containing the vertex key which uniquely
        identifies the vertex in the edge table.

        Defaults to None.
    object_type_as_bin : bool, optional
        If True, the object type will be considered CLOB in SAP HANA.

        Defaults to False.
    drop_exist_tab : bool, optional
        If force is True, drop the existing table when drop_exist_tab is
        True and truncate the existing table when it is
        False.

        Defaults to False.
    allow_bigint : bool, optional
        allow_bigint decides whether int64 is mapped into INT or BIGINT in HANA.

        Defaults to False.
    force_tables : bool, optional
        If force_tables is True, then the SAP HANA tables for vertices and edges
         are truncated or dropped.

        Defaults to False.
    force_workspace : bool, optional
        If force_workspace is True, then an existing workspace is overwritten
        during the creation process.

        Defaults to False.
    replace : bool, optional
        If replace is True, then the SAP HANA table performs the missing
        value handling.

        Defaults to True.
    geo_cols : list, optional but required for spatial functions with Pandas dataframes
        Specifies the columns of the Pandas dataframe, which are treated as geometries.
        List elements can be either strings or tuples.

        The geo_cols will be tested against columns in the vertices and edges
        dataframes. Depending on the existance, they will be distributed to the
        according table. geo_cols, that don't exist in either dataframe will
        be ignored. The `srid` applies to both dataframes.

        If you need a more deliberate management, consider to transform the Pandas
        dataframes with :func:`create_dataframe_from_pandas` to HANA dataframes first.
        Here you can specificaly control the tranformation according to the function
        features.

        **Strings** represent columns which contain geometries in (E)WKT format.
        If the provided DataFrame is a GeoPandas DataFrame, you do not need
        to add the geometry column to the geo_cols. It will be detected and
        added automatically.

        The column name in the HANA Table will be `<column_name>_GEO`


        **Tuples** must consist of two or strings: `(<longitude column>, <latitude column>)`

        `longitude column`: Dataframe column, that contains the longitude values

        `latitude column`: Dataframe column, that contains the latitude values

        They will be combined to a `POINT(<longiturd> <latitude>`) geometry.

        The column name in the HANA Table will be `<longitude>_<latitude>_GEO`

        Defaults to None.
    srid : int, optional but required for spatial functions with Pandas dataframes
        Spatial reference system id.

        Defaults to 4326.

    Returns
    -------
    Graph
        A virtual HANA Graph with functions inherited from the individual
        vertex and edge HANA Dataframes.

    Examples
    --------
    >>> e_pdf = pd.read_csv(self.e_path)
    >>>
    >>> hg = create_graph_from_edges_dataframe(
    >>>     connection_context=self._connection_context,
    >>>     edges_df=e_pdf,
    >>>     workspace_name="factory_ws",
    >>>     edge_source_column="from",
    >>>     edge_target_column="to",
    >>>     edge_key_column="edge_id",
    >>>     drop_exist_tab=True,
    >>>     force_tables=True,
    >>>     force_workspace=True,
    >>> )
    >>>
    >>> print(hg)
    """
    if not schema:
        schema = connection_context.get_current_schema()

    delete_temp_key = False
    if isinstance(edges_df, pd.DataFrame):
        # If there is no edge_col_key then assign one called EDGE_ID and base
        # values on a row sequence for id
        if not edge_key_column:
            edge_key_column = EDGE_ID
            delete_temp_key = True
            edges_df.insert(loc=0, column=EDGE_ID, value=np.arange(len(edges_df)))

        # Create the Edge table within the same schema but not as its own Dataframe
        edges_hdf = create_dataframe_from_pandas(
            connection_context,
            pandas_df=edges_df,
            table_name="{}_EDGES".format(workspace_name),
            schema=schema,
            force=force_tables,
            replace=replace,
            object_type_as_bin=object_type_as_bin,
            drop_exist_tab=drop_exist_tab,
            allow_bigint=allow_bigint,
            geo_cols=geo_cols,
            srid=srid,
            primary_key=edge_key_column,
            not_nulls=[edge_key_column, edge_source_column, edge_target_column],
        )
    elif isinstance(edges_df, DataFrame):
        if not edge_key_column:
            raise ValueError(
                "For hana_ml.DataFrames, an `edge_key_column` needs to be provided"
            )

        edges_hdf = edges_df
    else:
        raise ValueError(
            "The edges dataframe mus be either a HANA or a Pandas dataframe."
        )

    # Create Vertices View
    sql = """SELECT "{}" AS VERTEX_ID
               FROM ({})
              UNION
             SELECT "{}" AS VERTEX_ID
               FROM ({})""".format(
        edge_source_column,
        edges_hdf.select_statement,
        edge_target_column,
        edges_hdf.select_statement,
    )
    vertices_hdf = DataFrame(
        connection_context=connection_context, select_statement=sql
    )

    # Delegate
    hg = create_graph_from_hana_dataframes(
        connection_context=connection_context,
        vertices_df=vertices_hdf,
        vertex_key_column="VERTEX_ID",
        edges_df=edges_hdf,
        edge_key_column=edge_key_column,
        workspace_name=workspace_name,
        schema=schema,
        edge_source_column=edge_source_column,
        edge_target_column=edge_target_column,
        force=force_workspace,
    )

    # Drop the column we added. We don't want to change the df to the outside
    # caller
    if delete_temp_key:
        edges_df.drop(columns=EDGE_ID, inplace=True)

    return hg


def create_graph_from_dataframes(
    connection_context: ConnectionContext,
    vertices_df,
    vertex_key_column: str,
    edges_df,
    workspace_name: str,
    schema: str = None,
    edge_source_column: str = "from",
    edge_target_column: str = "to",
    edge_key_column: str = None,
    object_type_as_bin: bool = False,
    drop_exist_tab: bool = True,
    allow_bigint: bool = False,
    force_tables: bool = True,
    force_workspace: bool = True,
    replace: bool = False,
    geo_cols: list = None,  # Spatial variable
    srid: int = 4326,  # Spatial variable
) -> "Graph":
    """
    Create a HANA Graph workspace based on an edge and a vertices dataframe. The respective
    vertices table is created implicitely based on the `from` and `to` columns
    of the edges.

    Expects either HANA dataframes or pandas dataframes as input.
    If they are pandas then they will be transformed into `hana_ml.DataFrame`.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    vertices_df : pandas.DataFrame or hana_ml.DataFrame
        Table of data containing vertices and their keys that correspond
        with the edge frame.
    edges_df : pandas.DataFrame or hana_ml.DataFrame
        Table of data containing edges that link keys within the vertex frame.
    workspace_name : str
        Name of the workspace expected in the SAP HANA Graph workspaces
        of the ConnectionContext.
    schema : str
        Schema name of the workspace. If this value is not provided or set to
        None, then the value defaults to the ConnectionContext's current schema.

        Defaults to the current schema.
    edge_source_column : str
        Column name in the e_frame containing only source vertex keys that
        exist within the vertex_key_column of the n_frame.

        Defaults to 'from'.
    edge_target_column : str
        Column name in the e_frame containing the unique id of the edge.

        Defaults to 'to'.
    edge_key_column : str
        Column name in the n_frame containing the vertex key which uniquely
        identifies the vertex in the edge table.

        Defaults to None.
    vertex_key_column : str
        Column name in the n_frame containing the vertex key which uniquely
        identifies the vertex in the edge table.

        Defaults to None.
    object_type_as_bin : bool, optional
        If True, the object type will be considered CLOB in SAP HANA.

        Defaults to False.
    drop_exist_tab : bool, optional
        If force is True, drop the existing table when drop_exist_tab is
        True and truncate the existing table when it is
        False.

        Defaults to False.
    allow_bigint : bool, optional
        allow_bigint decides whether int64 is mapped into INT or BIGINT in HANA.

        Defaults to False.
    force_tables : bool, optional
        If force_tables is True, then the SAP HANA tables for vertices and edges
         are truncated or dropped.

        Defaults to False.
    force_workspace : bool, optional
        If force_workspace is True, then an existing workspace is overwritten
        during the creation process.

        Defaults to False.
    replace : bool, optional
        If replace is True, then the SAP HANA table performs the missing
        value handling.

        Defaults to True.
    geo_cols : list, optional but required for spatial functions with Pandas dataframes
        Specifies the columns of the Pandas dataframe, which are treated as geometries.
        List elements can be either strings or tuples.

        The geo_cols will be tested against columns in the vertices and edges
        dataframes. Depending on the existance, they will be distributed to the
        according table. geo_cols, that don't exist in either dataframe will
        be ignored. The `srid` applies to both dataframes.

        If you need a more deliberate management, consider to transform the Pandas
        dataframes with :func:`create_dataframe_from_pandas` to HANA dataframes first.
        Here you can specificaly control the tranformation according to the function
        features.

        **Strings** represent columns which contain geometries in (E)WKT format.
        If the provided DataFrame is a GeoPandas DataFrame, you do not need
        to add the geometry column to the geo_cols. It will be detected and
        added automatically.

        The column name in the HANA Table will be `<column_name>_GEO`


        **Tuples** must consist of two or strings: `(<longitude column>, <latitude column>)`

        `longitude column`: Dataframe column, that contains the longitude values

        `latitude column`: Dataframe column, that contains the latitude values

        They will be combined to a `POINT(<longiturd> <latitude>`) geometry.

        The column name in the HANA Table will be `<longitude>_<latitude>_GEO`

        Defaults to None.
    srid : int, optional but required for spatial functions with Pandas dataframes
        Spatial reference system id.

        Defaults to 4326.

    Returns
    -------
    Graph
        A virtual HANA Graph with functions inherited from the individual
        vertex and edge HANA Dataframes.

    Examples
    --------
    >>> v_pdf = pd.read_csv("nodes.csv")
    >>> e_pdf = pd.read_csv("edges.csv")
    >>>
    >>> hg = create_graph_from_dataframes(
    >>>     self._connection_context,
    >>>     vertices_df=v_pdf,
    >>>     edges_df=e_pdf,
    >>>     workspace_name="test_factory_ws",
    >>>     vertex_key_column="guid",
    >>>     geo_cols=[("lon", "lat")],
    >>>     force_tables=True,
    >>>     force_workspace=True,
    >>> )
    >>>
    >>> print(hg)
    """
    if not schema:
        schema = connection_context.get_current_schema()

    if isinstance(vertices_df, pd.DataFrame) and isinstance(edges_df, pd.DataFrame):
        # Delegate
        return _create_graph_from_pandas_dataframes(
            connection_context=connection_context,
            vertices_df=vertices_df,
            vertex_key_column=vertex_key_column,
            edges_df=edges_df,
            workspace_name=workspace_name,
            schema=schema,
            edge_key_column=edge_key_column,
            edge_source_column=edge_source_column,
            edge_target_column=edge_target_column,
            object_type_as_bin=object_type_as_bin,
            drop_exist_tab=drop_exist_tab,
            allow_bigint=allow_bigint,
            force_tables=force_tables,
            force_workspace=force_workspace,
            replace=replace,
            geo_cols=geo_cols,
            srid=srid,
        )
    elif isinstance(vertices_df, DataFrame) and isinstance(edges_df, DataFrame):
        if not edge_key_column:
            raise ValueError(
                "For hana_ml.DataFrames, an `edge_key_column` needs to be provided"
            )

        # Delegate
        return create_graph_from_hana_dataframes(
            connection_context=connection_context,
            vertices_df=vertices_df,
            vertex_key_column=vertex_key_column,
            edges_df=edges_df,
            edge_key_column=edge_key_column,
            workspace_name=workspace_name,
            schema=schema,
            edge_source_column=edge_source_column,
            edge_target_column=edge_target_column,
            force=force_workspace,
        )
    else:
        raise ValueError(
            "An edges and vertices definition are required. Both need to be of the same type."
        )


def _create_graph_from_pandas_dataframes(
    connection_context: ConnectionContext,
    vertices_df: pd.DataFrame,
    vertex_key_column: str,
    edges_df: pd.DataFrame,
    workspace_name: str,
    schema=None,
    edge_key_column: str = None,
    edge_source_column: str = "from",
    edge_target_column: str = "to",
    object_type_as_bin: bool = False,
    drop_exist_tab: bool = True,
    allow_bigint: bool = False,
    force_tables: bool = True,
    force_workspace: bool = True,
    replace: bool = False,
    geo_cols: list = None,  # Spatial variable
    srid: int = 4326,  # Spatial variable
):
    """
    Internal helper for handling pandas dataframes.
    :param connection_context:
    :param vertices_df:
    :param vertex_key_column:
    :param edges_df:
    :param workspace_name:
    :param schema:
    :param edge_key_column:
    :param edge_source_column:
    :param edge_target_column:
    :param object_type_as_bin:
    :param drop_exist_tab:
    :param allow_bigint:
    :param force_tables:
    :param force_workspace:
    :param replace:
    :param geo_cols:
    :param srid:
    :return:
    """

    def _get_geo_cols_for_df(geo_cols, df):
        cols = []
        for col in geo_cols:
            if isinstance(col, tuple):  # Check individual tuple columns
                if (col[0] in df.columns) != (col[1] in df.columns):
                    raise ValueError(
                        "Both columns of '{}' must be in '{}'".format(col, df.columns)
                    )
                elif col[0] in df.columns and col[0] in df.columns:
                    cols.append(col)
            else:
                if col in df.columns:
                    cols.append(col)

        return cols

    if not schema:
        schema = connection_context.get_current_schema()

    # Use the 2 pandas dataframes as vertex and edges tables to create a workspace
    # and return the collect statements
    vertex_table_name = "{}_VERTICES".format(workspace_name)
    edge_table_name = "{}_EDGES".format(workspace_name)

    # Check geo_cols and whether it references a column in the node or edges table
    if geo_cols is None:
        v_geo_cols = None
        e_geo_cols = None
    else:
        v_geo_cols = _get_geo_cols_for_df(geo_cols, vertices_df)
        e_geo_cols = _get_geo_cols_for_df(geo_cols, edges_df)

    vertices_hdf = create_dataframe_from_pandas(
        connection_context,
        pandas_df=vertices_df,
        table_name=vertex_table_name,
        schema=schema,
        force=force_tables,
        replace=replace,
        object_type_as_bin=object_type_as_bin,
        drop_exist_tab=drop_exist_tab,
        allow_bigint=allow_bigint,
        geo_cols=v_geo_cols,
        srid=srid,
        primary_key=vertex_key_column,
    )

    # If there is no edge_col_key then assign one called EDGE_ID and base
    # values on a row sequence for id
    delete_temp_key = False
    if not edge_key_column:
        edge_key_column = EDGE_ID
        delete_temp_key = True
        edges_df.insert(loc=0, column=EDGE_ID, value=np.arange(len(edges_df)))

    # Create the Edge table within the same schema but not as its own Dataframe
    edges_hdf = create_dataframe_from_pandas(
        connection_context,
        pandas_df=edges_df,
        table_name=edge_table_name,
        schema=schema,
        force=force_tables,
        replace=replace,
        drop_exist_tab=drop_exist_tab,
        object_type_as_bin=object_type_as_bin,
        allow_bigint=allow_bigint,
        geo_cols=e_geo_cols,
        srid=srid,
        primary_key=edge_key_column,
        not_nulls=[edge_key_column, edge_source_column, edge_target_column],
    )

    # Delegate
    hg = create_graph_from_hana_dataframes(
        connection_context=connection_context,
        vertices_df=vertices_hdf,
        vertex_key_column=vertex_key_column,
        edges_df=edges_hdf,
        edge_key_column=edge_key_column,
        workspace_name=workspace_name,
        schema=schema,
        edge_source_column=edge_source_column,
        edge_target_column=edge_target_column,
        force=force_workspace,
    )

    # Drop the column we added. We don't want to change the df to the outside
    # caller
    if delete_temp_key:
        edges_df.drop(columns=EDGE_ID, inplace=True)

    return hg


def create_graph_from_hana_dataframes(
    connection_context: ConnectionContext,
    vertices_df: DataFrame,
    vertex_key_column: str,
    edges_df: DataFrame,
    edge_key_column: str,
    workspace_name: str,
    schema: str = None,
    edge_source_column: str = "from",
    edge_target_column: str = "to",
    force: bool = False,
) -> "Graph":
    """
    Creates a graph workspace based on HANA DataFrames. This method can be
    used, uf some features are required, which are not provided in the
    :func:`create_graph_from_dataframes` (e.g. you need to set a chunk_size,
    when transferring the Pandas DataFrame to an HANA DataFrame. This is not
    offered in :func:`create_graph_from_dataframes`

    Based on the input dataframes the following logic applies for
    creating/selecting the source catalog objects from HANA for the graph workspace:

    * If both dataframes are based on database tables, both have a valid key
      column and the source and target columns in the vertices table are not
      nullable, then the graph workspace is based on the tables directly
    * If one of the tables does not fullfill above's criteria, or if at least
      one of the dataframes is based on a view or a SQL statement, respective
      views (on a table or an SQL view) are genrated, which will be used as
      a base for the graph workspace

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    vertices_df : hana_ml.DataFrame
        Table of data containing vertices and their keys that correspond
        with the edge frame.
    vertex_key_column : str
        Column name in the n_frame containing the vertex key which uniquely
        identifies the vertex in the edge table.
    edges_df : hana_ml.DataFrame
        Table of data containing edges that link keys within the vertex frame.
    edge_key_column : str
        Column name in the n_frame containing the vertex key which uniquely
        identifies the vertex in the edge table.
    workspace_name : str
        Name of the workspace expected in the SAP HANA Graph workspaces
        of the ConnectionContext.
    schema : str
        Schema name of the workspace. If this value is not provided or set to
        None, then the value defaults to the ConnectionContext's current schema.

        Defaults to the current schema.
    edge_source_column : str
        Column name in the e_frame containing only source vertex keys that
        exist within the vertex_key_column of the n_frame.

        Defaults to 'from'.
    edge_target_column : str
        Column name in the e_frame containing the unique id of the edge.

        Defaults to 'to'.
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
    >>> v_df = create_dataframe_from_pandas(
    >>>     connection_context=connection_context,
    >>>     pandas_df=pd.read_csv('nodes.csv'),
    >>>     table_name="factory_test_table_vertices",
    >>>     force=True,
    >>>     primary_key="guid",
    >>> )
    >>>
    >>> e_df = create_dataframe_from_pandas(
    >>>     connection_context=connection_context,
    >>>     pandas_df=pd.read_csv('edges.csv'),
    >>>     table_name="factory_test_table_edges",
    >>>     force=True,
    >>>     primary_key="edge_id",
    >>>     not_nulls=["from", "to"],
    >>> )
    >>>
    >>> hg = create_graph_from_hana_dataframes(
    >>>     connection_context=connection_context,
    >>>     vertices_df=v_df,
    >>>     vertex_key_column="guid",
    >>>     edges_df=e_df,
    >>>     edge_key_column="edge_id",
    >>>     workspace_name="test_factory_ws",
    >>>     force=True,
    >>> )
    >>>
    >>> print(hg)
    """

    def _count_views(
        cc: ConnectionContext, vertices_df: DataFrame, edges_df: DataFrame
    ):
        """ Counts how many of the data frames are based on views 0, 1, or 2 """
        sql = """
            SELECT COUNT(*) FROM SYS.VIEWS 
             WHERE SCHEMA_NAME='{}' AND VIEW_NAME='{}' 
                or SCHEMA_NAME='{}' AND VIEW_NAME='{}'
        """.format(
            vertices_df.source_table["SCHEMA_NAME"],
            vertices_df.source_table["TABLE_NAME"],
            edges_df.source_table["SCHEMA_NAME"],
            edges_df.source_table["TABLE_NAME"],
        )

        with cc.connection.cursor() as cur:
            cur.execute(sql)
            meta = cur.fetchall()

        return meta[0][0]

    def _get_table_metadata(cc: ConnectionContext, df: DataFrame):
        """ Helper to get table metadata """
        sql = """
            SELECT tc.COLUMN_NAME, c.IS_PRIMARY_KEY, tc.IS_NULLABLE FROM SYS.TABLE_COLUMNS AS tc 
                LEFT JOIN SYS.CONSTRAINTS AS c
                  ON tc.SCHEMA_NAME  = c.SCHEMA_NAME 
                 AND tc.TABLE_NAME  = c.TABLE_NAME 
                 AND tc.COLUMN_NAME = c.COLUMN_NAME 
            WHERE tc.SCHEMA_NAME = '{}' 
              AND tc.TABLE_NAME = '{}'
        """.format(
            df.source_table["SCHEMA_NAME"], df.source_table["TABLE_NAME"]
        )

        with cc.connection.cursor() as cur:
            cur.execute(sql)
            meta = cur.fetchall()
            meta_cols = [col[0] for col in meta]

        return dict(zip(meta_cols, meta))

    # Make sure, we always have a schema
    if not schema:
        schema = connection_context.get_current_schema()

    # Check if column names for keys are valid and source/target columns are valid
    if vertex_key_column not in vertices_df.columns:
        raise ValueError(
            "'{}' is not a column in the vertices data frame".format(vertex_key_column)
        )

    if edge_key_column not in edges_df.columns:
        raise ValueError(
            "'{}' is not a column in the edges data frame".format(edge_key_column)
        )

    if edge_source_column not in edges_df.columns:
        raise ValueError(
            "'{}' is not a column in the edges data frame".format(edge_source_column)
        )

    if edge_target_column not in edges_df.columns:
        raise ValueError(
            "'{}' is not a column in the edges data frame".format(edge_target_column)
        )

    # If both dataframes are based on tables, check if they can be used directly
    #   (i.e. table keys available and correct set and not nullable metadata are correct).
    # If both are based on a view, use the views directly. Constraints are then
    #   checked during Graph runtime
    # If one is based on a view and one on a table, we create two new views
    # If one or both are based on SQL statements, we create two new views
    create_views = False  # Assume, that we can use the DFs directly

    if (
        getattr(vertices_df, "source_table", None) is not None
        and getattr(edges_df, "source_table", None) is not None
    ):
        view_count = _count_views(connection_context, vertices_df, edges_df)

        if view_count == 1:  # If only one is a view -> create views for both
            create_views = True
        elif view_count == 2:  # If both are views -> use them directly
            create_views = False
        else:  # Both have `source_table` and are not views -> they must be tables
            vertices_metadata = _get_table_metadata(connection_context, vertices_df)
            edges_metadata = _get_table_metadata(connection_context, edges_df)

            # Check the primary keys
            is_key = edges_metadata.get(edge_key_column)[1]
            if is_key != "TRUE":
                create_views = True
                logger.debug(
                    "'%s' is not a primary key in the edge table", edge_key_column
                )

            is_key = vertices_metadata.get(vertex_key_column)[1]
            if is_key != "TRUE":
                create_views = True
                logger.debug(
                    "'%s' is not a primary key in the vertices table",
                    vertex_key_column,
                )

            # Check if Source and Target columns are not nullable
            is_nullable = edges_metadata.get(edge_source_column)[2]
            if is_nullable == "TRUE":
                create_views = True
                logger.debug("Edge source column is nullable")

            is_nullable = edges_metadata.get(edge_target_column)[2]
            if is_nullable == "TRUE":
                create_views = True
                logger.debug("Edge target column is nullable")

        # Define the sources for the Graph based on the criteria above
        if not create_views:  # Directly use set `source_table` (views or tables)
            edge_tbl_name = edges_df.source_table["TABLE_NAME"]
            vertex_tbl_name = vertices_df.source_table["TABLE_NAME"]
        else:  # Create new views for both dataframes
            edge_tbl_name = "{}_GE_VIEW".format(edges_df.source_table["TABLE_NAME"])
            vertex_tbl_name = "{}_GV_VIEW".format(
                vertices_df.source_table["TABLE_NAME"]
            )
    else:  # At least one is based on a SQL statement -> create views for both
        create_views = True

        edge_tbl_name = "{}_GE_VIEW".format(workspace_name)
        vertex_tbl_name = "{}_GV_VIEW".format(workspace_name)

    if create_views:
        vertices_df.save(where=(schema, vertex_tbl_name), table_type="VIEW", force=True)
        logger.info("Vertex view '%s' for graph created", vertex_tbl_name)

        edges_df.save(where=(schema, edge_tbl_name), table_type="VIEW", force=True)
        logger.info("Edge view '%s' for graph created", edge_tbl_name)

    logger.info("Graph is based on '%s' and '%s'", edge_tbl_name, vertex_tbl_name)

    if force:
        try:
            connection_context.connection.cursor().execute(
                """DROP GRAPH WORKSPACE "{}"."{}" """.format(schema, workspace_name)
            )
        except dbapi.Error as error:
            if "invalid graph workspace name:" in error.errortext:
                pass
            else:
                logger.error(error.errortext)
                raise

    # Create the graph workspace
    sql = """
            CREATE GRAPH WORKSPACE "{schema}"."{workspace_name}"
            EDGE TABLE "{schema}"."{edge_table}"
            SOURCE COLUMN "{source_column}"
            TARGET COLUMN "{target_column}"
            KEY COLUMN "{edge_id}"
            VERTEX TABLE "{schema}"."{vertex_table}"
            KEY COLUMN "{vertex_key_column}"
        """.format(
        schema=schema,
        workspace_name=workspace_name,
        edge_table=edge_tbl_name,
        source_column=edge_source_column,
        target_column=edge_target_column,
        edge_id=edge_key_column,
        vertex_table=vertex_tbl_name,
        vertex_key_column=vertex_key_column,
    )

    try:
        connection_context.connection.cursor().execute(sql)
    except dbapi.Error as error:
        logger.error(error.errortext)
        raise

    # Import the Graph Object here, to avoid circular dependencies
    from .hana_graph import Graph  # pylint: disable=import-outside-toplevel

    return Graph(
        connection_context=connection_context,
        workspace_name=workspace_name,
        schema=schema,
    )
