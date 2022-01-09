"""
This module represents a database query as a dataframe.
Most operations are designed to not bring data back from the database
unless explicitly requested.

The following classes and functions are available:

    * :class:`ConnectionContext`
    * :class:`DataFrame`
    * :func:`quotename`
    * :func:`create_dataframe_from_pandas`
    * :func:`create_dataframe_from_shapefile`
    * :func:`read_pickle`
    * :func:`melt`
"""
#pylint: disable=too-many-lines
#pylint: disable=line-too-long
#pylint: disable=relative-beyond-top-level
#pylint: disable=fixme
#pylint: disable=too-many-locals
#pylint: disable=too-many-branches
#pylint: disable=deprecated-lambda
#pylint: disable=too-many-arguments
#pylint: disable=too-many-format-args
#pylint: disable=too-many-statements
#pylint: disable=bare-except
#pylint: disable=broad-except
#pylint: disable=singleton-comparison
#pylint: disable=deprecated-method
#pylint: disable=protected-access
#pylint: disable=too-many-nested-blocks
#pylint: disable=redefined-outer-name
#pylint: disable=no-self-use
#pylint: disable=consider-using-f-string
import logging
import sys
import uuid
import os
import math
import re
from zipfile import ZipFile
import getpass
import numpy as np
import pandas as pd
from hdbcli import dbapi
from tqdm import tqdm
try:
    from shapely import wkt
except ImportError as error:
    logging.getLogger(__name__).error("%s: %s", error.__class__.__name__, str(error))
    pass
try:
    import pyodbc
except ImportError as error:
    pass
from .ml_exceptions import BadSQLError
from .ml_base import execute_logged
from .type_codes import get_type_code_map

TYPE_CODES = get_type_code_map()
logger = logging.getLogger(__name__) #pylint: disable=invalid-name

if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INTEGER_TYPES = (int, long)
    _STRING_TYPES = (str, unicode)
else:
    _INTEGER_TYPES = (int,)
    _STRING_TYPES = (str,)


def quotename(name):
    """
    Escapes a schema, table, or column name for use in SQL. hana_ml functions and methods that take schema, table, or column names
    already escape their input by default, but those that take SQL don't (and can't) perform escaping automatically.

    Parameters
    ----------
    name : str
        The schema, table, or column name.

    Returns
    -------
    str
        The escaped name. The string is surrounded in quotation marks,
        and existing quotation marks are escaped by doubling them.
    """

    return '"{}"'.format(name.replace('"', '""'))

class ConnectionContext(object):
    """
    Represents a connection to the SAP HANA system.

    ConnectionContext includes methods for creating DataFrames from data
    on the SAP HANA. DataFrames are tied to a ConnectionContext, and are unusable
    once their ConnectionContext is closed.

    Parameters
    ----------
    Same as hdbcli.dbapi.connect.

        Please see the `online docs for hdbcli.dbapi.connect\
        <https://help.sap.com/viewer/0eec0d68141541d1b07893a39944924e/latest/en-US/ee592e89dcce4480a99571a4ae7a702f.html>`_
        for more details.

    Examples
    --------
    Querying data from SAP HANA into a Pandas DataFrame:

    >>> with ConnectionContext('address', port, 'user', 'password') as cc:
    ...     df = (cc.table('MY_TABLE', schema='MY_SCHEMA')
    ...             .filter('COL3 > 5')
    ...             .select('COL1', 'COL2'))
    ...     pandas_df = df.collect()

    The underlying hdbcli.dbapi.connect can be accessed if necessary:

    >>> with ConnectionContext('127.0.0.1', 30215, 'MLGUY', 'manager') as cc:
    ...     cc.connection.setclientinfo('SOMEKEY', 'somevalue')
    ...     df = cc.sql('some sql that needs that session variable')
    ...     ...

    Attributes
    ----------
    connection : hdbcli.dbapi.connect
        The underlying dbapi connection. Use this connection to run SQL directly,
        or to access connection methods like getclientinfo/setclientinfo.
    """

    def __init__(self,            #pylint: disable=too-many-arguments
                 address='',
                 port=0,
                 user='',
                 password=None,
                 autocommit=True,
                 packetsize=None,
                 userkey=None,
                 spatialtypes=1,
                 encrypt=None,
                 sslValidateCertificate=None,
                 pyodbc_connection=None,
                 **properties):
        key = None
        if 'key' in properties:
            key = properties.pop('key')
        if not pyodbc_connection:
            if password is None and ((userkey is None) and (key is None)):
                while True:
                    password = getpass.getpass("HANA DB User : %s Password : " % user)
                    if password is None:
                        password = ''
                        break
                    if password is not None:
                        break
        if password is None:
            password = ''

        if str(spatialtypes) != '1':
            logger.warning("With 'spatialtypes=%s', this connection does not support spatial features in the dataframe", spatialtypes)
        self.connection = None
        self.pyodbc_connection = pyodbc_connection
        if pyodbc_connection:
            self.connection = pyodbc.connect(pyodbc_connection)
        else:
            self.connection = dbapi.connect(
                address,
                port,
                user,
                password,
                autocommit=autocommit,
                packetsize=packetsize,
                userkey=userkey,
                spatialtypes=spatialtypes,
                encrypt=encrypt,
                sslValidateCertificate=sslValidateCertificate,
                **properties)
        self._pal_check_passed = False
        self.sql_tracer = SqlTrace() # SQLTRACE
        self.last_execute_statement = None

    def close(self):
        """
        Closes the existing connection to the SAP HANA system.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        cc is a connection to SAP HANA system.

        >>> cc.close()
        """

        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def add_primary_key(self, table, columns, schema=None):
        """
        Add primary key to the existing table.

        Parameters
        ----------
        table : str
            Table name.

        columns : str of list
            Columns to be primary key.

        schema : str, optional
            Schema name. If None, use the current schema.
        """

        if schema is None:
            table_reference = quotename(table)
        else:
            table_reference = '.'.join(map(quotename, (schema, table)))
        column_str = ''
        if isinstance(columns, str):
            column_str = columns
        else:
            column_str = ', '.join(columns)
        query = "ALTER TABLE {} ADD PRIMARY KEY ({})".format(table_reference, quotename(column_str))
        with self.connection.cursor() as cur:
            cur.execute(query)

    def create_table(self, table, table_structure, schema=None, table_type='COLUMN', prop='', data_lake=False, data_lake_container='SYSRDL#CG'):
        """
        Create a SAP HANA table.

        Parameters
        ----------
        table : str
            Table name.
        table_structure : dict
            SAP HANA table structure. {Column name: Column type, ...}
        schema : str, optional
            Schema name. If None, use the current schema.
        table_type : str, optional
            Specify the table type.

            - 'COLUMN', by default.
            - 'ROW'.
            - 'TEMPORARY'.
        data_lake : bool, optional
            If True, create the data lake table by suing SYSRDL#CG.REMOTE_EXECUTE().

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> cc.create_table(table='TEST_TBL', table_structure={'test_name': 'VARCHAR(50)'})
        """
        if schema is None:
            table_reference = quotename(table)
        else:
            table_reference = '.'.join(map(quotename, (schema, table)))
        t_type = ' '
        if table_type.upper() == 'COLUMN' and (not table.startswith("#")):
            t_type = ' COLUMN '
            if data_lake:
                t_type = ' '
        elif table_type.upper() == 'TEMPORARY' or table.startswith("#"):
            t_type = ' LOCAL TEMPORARY '
        else:
            t_type = ' ' + table_type + ' '
        query = 'CREATE{}TABLE {} ('.format(t_type, table_reference)
        for key, val in table_structure.items():
            query = query + "{} {}, ".format(quotename(key), val)
        query = query[:-2] + ") {}".format(prop)
        if data_lake:
            query = "CALL {}.REMOTE_EXECUTE ('{}')".format(data_lake_container, query)
        try:
            with self.connection.cursor() as cur:
                execute_logged(cur,
                               query,
                               self.sql_tracer,
                               self)
        except dbapi.Error as ct_err:
            logger.error("%s. Failed to execute `%s`", ct_err, query)
            raise
        except pyodbc.Error as ct_err:
            logger.error("%s. Failed to execute `%s`", str(ct_err.args[1]), query)
            raise

    def create_virtual_table(self, table, data_lake_table=None, schema=None, data_lake_container='SYSRDL#CG',
                             remote_source=None, remote_database='NULL', remote_schema=None, remote_table=None):
        """
        Create a SAP virtual HANA table at remote/data lake source.

        Parameters
        ----------
        table : str
            HANA virtual table name.
        data_lake_table : str, optional
            HANA data lake table name.

            Mandatory if ``remote_table`` is None.
        schema : str, optional
            Schema name. If None, use the current schema.
        data_lake_container : str, optional
            Name of the data lake container.

            Defaults to 'SYSRDL#CG'
        remote_source : str, optional
            Remote source where data of the target virtual table reside.

            Mandatory and valid only if ``data_lake_table`` is None.
        remote_database : str, optional
            Database of the remote source where data of the target virtual table reside.

            Defaults to 'NULL', i.e. the default database.

            Valid only if ``data_lake_table`` is None.
        remote_schema : str, optional
            The schema under which the corresponding ``remote_table`` of the target SAP HANA
            virtual table reside.

            Required if ``data_lake_table`` is None.

            Defaults to None.
        remote_table : str, optional
            The table name in remote source where data of the target virtual table reside.

            Mandatory and valid only when ``data_lake_table`` is None.

            Defaults to None.
        """
        if all(x is None for x in (data_lake_table, remote_table)):
            msg = '`data_lake_table` and `remote_table` cannot both be set to None.'
            raise ValueError(msg)
        if schema is None:
            table_reference = quotename(table)
        else:
            table_reference = '.'.join(map(quotename, (schema, table)))
        if data_lake_table is not None:
            sql = 'CREATE VIRTUAL TABLE {0} AT "{1}_SOURCE"."NULL"."{1}".{2}'.format(table_reference,
                                                                                     data_lake_container,
                                                                                     quotename(data_lake_table))
        else:
            if any(x is None for x in (remote_source, remote_schema, remote_table)):
                msg = '`remote_source`, `remote_schema` and `remote_table` must all be specified.'
                raise ValueError(msg)
            sql = 'CREATE VIRTUAL TABLE {0} AT "{1}"."{2}"."{3}"."{4}"'.format(table_reference,
                                                                               remote_source,
                                                                               remote_database,
                                                                               remote_schema,
                                                                               remote_table)
        try:
            with self.connection.cursor() as cur:
                execute_logged(cur,
                               sql,
                               self.sql_tracer,
                               self)
        except dbapi.Error as cvt_err:
            logger.error("%s. Failed to execute `%s`", cvt_err, sql)
            raise
        except pyodbc.Error as cvt_err:
            logger.error("%s. Failed to execute `%s`", str(cvt_err.args[1]), sql)
            raise

    def drop_table(self, table, schema=None, data_lake=False, data_lake_container='SYSRDL#CG'):
        """
        Drop the specified table.

        Parameters
        ----------
        table : str
            Table name.
        schema : str, optional
            Schema name. If None, use the current schema.
        data_lake : bool, optional
            If True, drop the data lake table.

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> cc.drop_table(table='TEST_TBL')
        """
        if schema is None:
            schema = self.get_current_schema()
        try:
            with self.connection.cursor() as cur:
                query = 'DROP TABLE {}.{}'.format(quotename(schema), quotename(table))
                if data_lake:
                    query = 'DROP TABLE {}'.format(quotename(table))
                    query = "CALL {}.REMOTE_EXECUTE ('{}')".format(data_lake_container, query)
                execute_logged(cur,
                               query,
                               self.sql_tracer,
                               self)
        except dbapi.Error as db_er:
            logger.error("Fail to drop table. %s", db_er)
            pass
        except pyodbc.Error as db_er:
            logger.error("Fail to drop table. %s", str(db_er.args[1]))
            pass

    def copy_to_data_lake(self, data, virtual_table, data_lake_table, schema=None, append=False, data_lake_container='SYSRDL#CG'):
        """
        Copy HANA data to data lake table.

        Parameters
        ----------
        data : DataFrame
            HANA DataFrame.
        virtual_table : str
            HANA virtual table name.
        data_lake_table : str
            HANA data lake table name.
        schema : str, optional
            Schema name. If None, use the current schema.
        append : bool, optional
            Append data to the existing data lake table.
        """
        if not append:
            table_structure = data.get_table_structure()
            for key, value in table_structure.items():
                if value == 'NCLOB':
                    table_structure[key] = 'CLOB'
                if value == 'NBLOB':
                    table_structure[key] = 'BLOB'
                if 'NVARCHAR' in value:
                    table_structure[key] = value.replace('NVARCHAR', 'VARCHAR')
            self.create_table(
                table=data_lake_table,
                table_structure=table_structure,
                schema=schema,
                data_lake=True,
                data_lake_container=data_lake_container
            )
            self.create_virtual_table(
                table=virtual_table,
                data_lake_table=data_lake_table,
                schema=schema,
                data_lake_container=data_lake_container)
        if schema is None:
            table_reference = quotename(virtual_table)
        else:
            table_reference = '.'.join(map(quotename, (schema, virtual_table)))
        sql = "INSERT INTO {} {}".format(table_reference, data.select_statement)
        with self.connection.cursor() as cur:
            execute_logged(cur,
                           sql,
                           self.sql_tracer,
                           self)

    def has_table(self, table, schema=None):
        """
        Returns the boolean value for the table existence.

        Parameters
        ----------
        table : str
            Table name.
        schema : str, optional
            Schema name. If None, use the current schema.

        Returns
        -------
        bool
            Table existence.

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> cc.has_table(table='TEST_TBL')
        True
        """
        if schema is None:
            schema = self.get_current_schema()
        cnt_tab = -1
        if '#' in table:
            connection_id = DataFrame(self, "SELECT SESSION_CONTEXT('CONN_ID') FROM DUMMY").collect().iat[0, 0]
            cnt_tab = DataFrame(self, "SELECT COUNT(*) FROM M_TEMPORARY_TABLES WHERE TABLE_NAME='{}' AND SCHEMA_NAME='{}' AND CONNECTION_ID='{}'"\
            .format(table, schema, connection_id)).collect().iat[0, 0]
        else:
            cnt_tab = DataFrame(self, "SELECT COUNT(*) FROM M_TABLES WHERE TABLE_NAME='{}' AND SCHEMA_NAME='{}'"\
            .format(table, schema)).collect().iat[0, 0]
        if isinstance(cnt_tab, (list, tuple)):
            return cnt_tab[0] > 0
        return cnt_tab > 0

    def hana_version(self):
        """
        Returns the version of SAP HANA system.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The SAP HANA version.

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> cc.hana_version()
        '4.50.000.00.1581545459 (master)'
        """

        return DataFrame(self, "SELECT VALUE FROM SYS.M_SYSTEM_OVERVIEW WHERE NAME='Version'").collect().iat[0, 0]

    def get_current_schema(self):
        """
        Returns the current schema name.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The current schema name.

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> cc.get_current_schema()
        'TEST'
        """
        current_schema = DataFrame(self, "SELECT CURRENT_SCHEMA FROM DUMMY").collect().iat[0, 0]
        if isinstance(current_schema, (list, tuple)):
            current_schema = current_schema[0]
        return current_schema

    def get_tables(self, schema=None):
        """
        Returns the table list given schema.

        Parameters
        ----------
        schema : str, optional
            The schema name. If no specified, use the current schema.
        """
        if schema is None:
            schema = self.get_current_schema()
        return self.sql("SELECT DISTINCT TABLE_NAME FROM M_TABLES WHERE SCHEMA_NAME='{}'".format(schema)).collect()

    def hana_major_version(self):
        """
        Returns the major number of SAP HANA version.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The major number of SAP HANA version.

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> cc.hana_major_version()
        '4'
        """

        return self.hana_version().split(".", 1)[0]

    def is_cloud_version(self):
        """
        Check whether the SAP HANA instance is cloud version or on-premise.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            If True, the HANA instance is cloud version.
        """

        return int(self.hana_major_version()) >= 4

    def read_onnx(self, onnx_file, table, schema=None, opset=7):
        """
        Read onnx file content and return HANA DataFrame.

        Parameters
        ----------
        onnx_file : str
            ONNX file location.

        table : str
            HANA table to persist onnx model.

        schema : str, optional
            Specifies the schema.

        opset : int, optional
            The opset of ONNX model.
        """
        self.create_table(table=table, table_structure={"OPSET": "INT", "MODEL": "BLOB"}, schema=schema)
        if os.path.isfile(onnx_file):
            text_file = open(onnx_file, "rb")
            data = text_file.read()
            text_file.close()
            with self.connection.cursor() as cur:
                if schema is None:
                    table_reference = quotename(table)
                else:
                    table_reference = '.'.join(map(quotename, (schema, table)))
                sql = 'INSERT INTO {} VALUES (?, ?)'.format(table_reference)
                self.last_execute_statement = sql
                cur.execute(sql, (opset, data))
                cur.close()
        return self.table(table, schema)

    def sql(self, sql):
        """
        Returns a SAP HANA DataFrame representing a query.

        Parameters
        ----------
        sql : str
            SQL query. The last sentence must be select statement.

        Returns
        -------
        DataFrame
            The DataFrame for the query.

        Examples
        --------
        cc is a connection to a SAP HANA system.

        >>> df = cc.sql('SELECT T.A, T2.B FROM T, T2 WHERE T.C=T2.C')
        """
        sql = sql.rstrip("\n")
        anblock = re.search(r'(do begin|DO BEGIN|DO\nBEGIN|do\nbegin)[\s\S]+(end;|END;)', sql)
        multiline = None
        if anblock:
            part1, part2 = sql.split(anblock.group(0))
            part1 = part1.split(";")
            if part1[-1] == "":
                part1 = part1[:-1]
            part2 = part2.split(";")
            if part2[-1] == "":
                part2 = part2[:-1]
            multiline = part1 + [anblock.group(0)] + part2
        else:
            multiline = sql.split(";")
        if multiline[-1] == '':
            multiline = multiline[:-1]
        if len(multiline) > 1:
            while '\n' in multiline:
                multiline.remove('\n')
            while '' in multiline:
                multiline.remove('')
            with self.connection.cursor() as cur:
                for line in multiline[:-1]:
                    try:
                        execute_logged(cur,
                                       line,
                                       self.sql_tracer,
                                       self)
                    except dbapi.Error as err:
                        logger.error(err)
                    except pyodbc.Error as err:
                        logger.error(err)
            if not self.pyodbc_connection:
                if not self.connection.getautocommit():
                    self.connection.commit()
        return DataFrame(self, multiline[-1])

    def execute_sql(self, sql):
        """
        Multiline sql execution.

        Parameters
        ----------
        sql : str or sql file
            SQL query.
        """
        if os.path.isfile(sql):
            with open(sql, "r") as text_file:
                data = text_file.read()
            sql = data
        last_execute = self.sql(sql).select_statement
        with self.connection.cursor() as cur:
            try:
                execute_logged(cur, last_execute, self.sql_tracer, self)
            except dbapi.Error as err:
                logger.error(err)
            except pyodbc.Error as err:
                logger.error(err)
        if not self.pyodbc_connection:
            if not self.connection.getautocommit():
                self.connection.commit()

    def table(self, table, schema=None, save_source=True, view_params=None):
        """
        Returns a DataFrame that represents the specified table.

        Parameters
        ----------
        table : str
            The table name.
        schema : str, optional, keyword-only
            The schema name. If this value is not provided or set to None, then the value defaults to the
            ConnectionContext's current schema.
        save_source : bool, optional
            If True, save the name of source table.
            Defaults to True.
        view_params : list or tuple, optional
            Parameters for view.

        Returns
        -------
        DataFrame
            The DataFrame that is selecting data from the specified table.

        Examples
        --------
        >>> df1 = cc.table('MY_TABLE')
        >>> df2 = cc.table('MY_OTHER_TABLE', schema='MY_SCHEMA')
        """

        if schema is None:
            table_reference = quotename(table)
        else:
            table_reference = '.'.join(map(quotename, (schema, table)))

        select = 'SELECT * FROM {}'.format(table_reference)
        if view_params:
            mod_params = []
            for elem in view_params:
                if isinstance(elem, str):
                    mod_params.append("'{}'".format(elem))
                else:
                    mod_params.append(str(elem))
            select = "{} ({})".format(select, ", ".join(mod_params))
        result = DataFrame(self, select)

        if save_source:
            result.set_source_table(table, schema)

        # pylint: disable=protected-access
        if table.startswith('#'):
            result._ttab_handling = 'ttab'
            result._ttab_reference = table_reference
        else:
            result._ttab_handling = 'safe'

        # SQLTRACE
        # Checking the trace_sql_active is unnecessary however as this is
        # done in the sql_tracer object as well. But as not to impact
        # current test cases this has been added. If the ehck is removed 2
        # test cases will fail. Changing the test cases would be the better
        # option going forward
        if self.sql_tracer.trace_sql_active:
            self.sql_tracer.trace_object({
                'name': table,
                'table_type': result.generate_table_type(),
                'select': result.select_statement,
                'reference': table_reference,
                'schema': schema
            }, sub_cat='output_tables')

        return result

# SQLTRACE Class for Core Functions
class SqlTrace(object):
    """
    Provides functions to track generated SQL.

    It stores the trace in a dictionary in the following format:

    {
        'algorithm':{
            'function':{
                'subcategory':[]
            }
        }
    }

    Attributes
    ----------
    trace_sql_log : dictionary
        The SQL Trace dictionary
    trace_sql_active : boolean
        Flag to define if tracing should occur
    trace_sql_algo : string
        Current algorithm that is being traced
    trace_sql_function : string
        Current function that is being traced
    trace_history : boolean
        Track multiple runs of the same algorithm. If this attribute is enabled, then
        the algo_tracker tracks the count of the algorithm
        and adds a sequence number to the algorithm name in the dictionary
        so that each run is traced seperately.
    trace_algo_tracker : dictionary
        If trace_history is enabled, then the algorithm tracks the count of the number of times
        the same algorithm is run

    Examples
    --------
    >>> Example snippet of the SQL trace dictionary:
    {
        "RandomForestClassifier": {
            "Fit": {
                "input_tables": [
                    {
                        "name": "#PAL_RANDOM_FOREST_DATA_TBL_0",
                        "type": "table (\"AccountID\" INT,\"ServiceType\" VARCHAR(21),\"ServiceName\" VARCHAR(14),\"DataAllowance_MB\" INT,\"VoiceAllowance_Minutes\" INT,\"SMSAllowance_N_Messages\" INT,\"DataUsage_PCT\" DOUBLE,\"DataUsage_PCT_PM\" DOUBLE,\"DataUsage_PCT_PPM\" DOUBLE,\"VoiceUsage_PCT\" DOUBLE,\"VoiceUsage_PCT_PM\" DOUBLE,\"VoiceUsage_PCT_PPM\" DOUBLE,\"SMSUsage_PCT\" DOUBLE,\"SMSUsage_PCT_PM\" DOUBLE,\"SMSUsage_PCT_PPM\" DOUBLE,\"Revenue_Month\" DOUBLE,\"Revenue_Month_PM\" DOUBLE,\"Revenue_Month_PPM\" DOUBLE,\"Revenue_Month_PPPM\" DOUBLE,\"ServiceFailureRate_PCT\" DOUBLE,\"ServiceFailureRate_PCT_PM\" DOUBLE,\"ServiceFailureRate_PCT_PPM\" DOUBLE,\"CustomerLifetimeValue_USD\" DOUBLE,\"CustomerLifetimeValue_USD_PM\" DOUBLE,\"CustomerLifetimeValue_USD_PPM\" DOUBLE,\"Device_Lifetime\" INT,\"Device_Lifetime_PM\" INT,\"Device_Lifetime_PPM\" INT,\"ContractActivityLABEL\" VARCHAR(8))",
                        "select": "SELECT \"AccountID\", \"ServiceType\", \"ServiceName\", \"DataAllowance_MB\", \"VoiceAllowance_Minutes\", \"SMSAllowance_N_Messages\", \"DataUsage_PCT\", \"DataUsage_PCT_PM\", \"DataUsage_PCT_PPM\", \"VoiceUsage_PCT\", \"VoiceUsage_PCT_PM\", \"VoiceUsage_PCT_PPM\", \"SMSUsage_PCT\", \"SMSUsage_PCT_PM\", \"SMSUsage_PCT_PPM\", \"Revenue_Month\", \"Revenue_Month_PM\", \"Revenue_Month_PPM\", \"Revenue_Month_PPPM\", \"ServiceFailureRate_PCT\", \"ServiceFailureRate_PCT_PM\", \"ServiceFailureRate_PCT_PPM\", \"CustomerLifetimeValue_USD\", \"CustomerLifetimeValue_USD_PM\", \"CustomerLifetimeValue_USD_PPM\", \"Device_Lifetime\", \"Device_Lifetime_PM\", \"Device_Lifetime_PPM\", \"ContractActivityLABEL\" FROM (SELECT a.* FROM #PAL_PARTITION_DATA_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C a inner join #PAL_PARTITION_RESULT_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C b        on a.\"AccountID\" = b.\"AccountID\" where b.\"PARTITION_TYPE\" = 1) AS \"DT_2\""
                    }
                ],
                "internal_tables":[
                    {
                        "name": "#PAL_RANDOM_FOREST_PARAM_TBL_0",
                        "type": [
                            [
                                "PARAM_NAME",
                                "NVARCHAR(5000)"
                            ],
                            [
                                "INT_VALUE",
                                "INTEGER"
                            ],
                            [
                                "DOUBLE_VALUE",
                                "DOUBLE"
                            ],
                            [
                                "STRING_VALUE",
                                "NVARCHAR(5000)"
                            ]
                        ]
                    }
                ],
                "output_tables":[
                    {
                        "name": "#PAL_RANDOM_FOREST_MODEL_TBL_0",
                        "type": "table (\"ROW_INDEX\" INT,\"TREE_INDEX\" INT,\"MODEL_CONTENT\" NVARCHAR(5000))",
                        "select": "SELECT * FROM \"#PAL_RANDOM_FOREST_MODEL_TBL_0\"",
                        "reference": "\"#PAL_RANDOM_FOREST_MODEL_TBL_0\"",
                        "schema": null
                    }
                ],
                "function":[
                    {
                        "name": "PAL_RANDOM_DECISION_TREES",
                        "schema": "_SYS_AFL",
                        "type": "pal"
                    }
                ],
                "sql":[
                    "DROP TABLE \"#PAL_RANDOM_FOREST_DATA_TBL_0\"",
                    "CREATE LOCAL TEMPORARY COLUMN TABLE \"#PAL_RANDOM_FOREST_DATA_TBL_0\" AS (SELECT \"AccountID\", \"ServiceType\", \"ServiceName\", \"DataAllowance_MB\", \"VoiceAllowance_Minutes\", \"SMSAllowance_N_Messages\", \"DataUsage_PCT\", \"DataUsage_PCT_PM\", \"DataUsage_PCT_PPM\", \"VoiceUsage_PCT\", \"VoiceUsage_PCT_PM\", \"VoiceUsage_PCT_PPM\", \"SMSUsage_PCT\", \"SMSUsage_PCT_PM\", \"SMSUsage_PCT_PPM\", \"Revenue_Month\", \"Revenue_Month_PM\", \"Revenue_Month_PPM\", \"Revenue_Month_PPPM\", \"ServiceFailureRate_PCT\", \"ServiceFailureRate_PCT_PM\", \"ServiceFailureRate_PCT_PPM\", \"CustomerLifetimeValue_USD\", \"CustomerLifetimeValue_USD_PM\", \"CustomerLifetimeValue_USD_PPM\", \"Device_Lifetime\", \"Device_Lifetime_PM\", \"Device_Lifetime_PPM\", \"ContractActivityLABEL\" FROM (SELECT a.* FROM #PAL_PARTITION_DATA_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C a inner join #PAL_PARTITION_RESULT_TBL_EC70569E_882B_11E9_9ACB_784F436CBD3C b        on a.\"AccountID\" = b.\"AccountID\" where b.\"PARTITION_TYPE\" = 1) AS \"DT_2\")",
                    "DROP TABLE \"#PAL_RANDOM_FOREST_PARAM_TBL_0\"",
                    "CREATE LOCAL TEMPORARY COLUMN TABLE \"#PAL_RANDOM_FOREST_PARAM_TBL_0\" (\n    \"PARAM_NAME\" NVARCHAR(5000),\n    \"INT_VALUE\" INTEGER,\n    \"DOUBLE_VALUE\" DOUBLE,\n    \"STRING_VALUE\" NVARCHAR(5000)\n)"
                ]
        }

    }
    """

    def __init__(self):
        self.trace_sql_log = {}
        self.trace_sql_active = False
        self.trace_sql_algo = None
        self.trace_sql_function = None
        self.trace_history = False
        self.trace_algo_tracker = {}

    def _set_log_level(self, logger, level):
        if level == 'info':
            logger.setLevel(logging.INFO)
        else:
            if level == 'warn':
                logger.setLevel(logging.WARN)
            else:
                if level == 'debug':
                    logger.setLevel(logging.DEBUG)
                else:
                    logger.setLevel(logging.ERROR)

    def set_log_level(self, level='info'):
        """
        Set logging level.

        Parameters
        ----------

        level : {'info', 'warn', 'debug', 'error'}
        """
        logging.basicConfig()
        for module in ["hana_ml.ml_base", 'hana_ml.dataframe', 'hana_ml.algorithms.pal']:
            logger = logging.getLogger(module)
            self._set_log_level(logger, level)

    def enable_sql_trace(self, enable):
        """
        Enables or disables the SQL trace

        Parameters
        ----------
        enable : boolean
            Enables or disables the SQL trace
        """
        if enable:
            self.set_log_level()

        else:
            self.set_log_level("warn")
        self.trace_sql_active = enable

    def set_sql_trace(self, algo_object=None, algo=None, function=None):
        """
        Activates the trace for a certain algorithm and function. Any subsequent calls are placed
        under the respective substructure (algorithm -> function) in the trace dictionary.

        The algo_oject is the actual algorithm instance object from a algorithm class. This object is used
        in case the history of multiple calls to the same algorithm must be traced to check if
        the same object is being used or not.

        Parameters
        ----------
        algo_object : object
            The actual algorithm object.
        algo : string
            The algorithm name
        function : string
            The algorithm function
        """

        if self.trace_history and algo_object:
            new_id = 0
            # Check if we already have run the algo based on the instance object.
            if str(id(algo_object)) in self.trace_algo_tracker:
                new_id = self.trace_algo_tracker[str(id(algo_object))]
            elif algo in self.trace_algo_tracker:
                new_id = self.trace_algo_tracker[algo] + 1
                self.trace_algo_tracker[algo] = new_id
                self.trace_algo_tracker[str(id(algo_object))] = new_id
            else:
                new_id += 1
                self.trace_algo_tracker[algo] = new_id
                self.trace_algo_tracker[str(id(algo_object))] = new_id
            algo += str(new_id)
        self.trace_sql_algo = algo
        self.trace_sql_function = function

    def clean_trace_history(self):
        """
        Clean the trace history.
        """
        self.trace_sql_log = {}

    def enable_trace_history(self, enable):
        """
        Enables the trace history on the algorithm level. This option allows for multiple calls to the same algorithm to be stored
        separately in the dictionary with a sequence number attached the algorithm name. This behavior
        is only available on the algorithm level. Using the same algorithm instance and calling the same function (such as Fit)
        twice would still overwrite the previous call.

        Parameters
        ----------
        enable : boolean
            Enables or disables the trace history.
        """

        self.trace_history = enable

    def get_sql_trace(self):
        """
        Returns the SQL trace dictionary.

        Returns
        -------
        dict
            A SQL Trace dictionary object.
        """

        return self.trace_sql_log

    def trace_sql(self, value=None):
        """
        Adds the SQL value to the current active algorithm and function in the SQL trace dictionary.

        Parameters
        ----------
        value : str
            The SQL entry.
        """

        self._trace_data(value, 'sql')

    def trace_object(self, value=None, sub_cat='nocat'):
        """
        Trace additional objects outside of the SQL entry. This option supports a more finegrained context
        and information than the SQL trace. For example, the input tables, output tables, and function
        being used for example. These are convenience objects to help you understand how the SQL is being structured. Generally
        speaking, these objects are dictionaries themselves and that is the current use case. However, it is not required to be.

        Parameters
        ----------
        value : object
            An ambiguous type of object that provides additional context outside of the SQL trace itself.
        sub_cat : str
            The subcategory or key that the value must be placed under. For example, 'output_tables'.

        """

        self._trace_data(value, sub_cat)

    def trace_sql_many(self, statement, param_entries=None):
        """
        Converts the insert executemany method on the hdbcli cursor object to multiple INSERT statements.
        This conversion ensures that only pure SQL is passed through.

        Parameters
        ----------
        statement : str
            The SQL statement.
        param_entries : list of tuples, or None
            The data in the INSERT statement.
        """

        if self.trace_sql_active:
            for param_entry in param_entries:
                processed_statement = statement
                for param_value in param_entry:
                    if isinstance(param_value, str):
                        processed_statement = processed_statement.replace('?', "'"+param_value+"'", 1)
                    else:
                        processed_statement = processed_statement.replace('?', str(param_value), 1)
                # Additional processing to assure proper SQL
                processed_statement = processed_statement.replace('None', 'null')
                processed_statement = processed_statement.replace('True', '1')
                processed_statement = processed_statement.replace('False', '0')
                self.trace_sql(processed_statement)

    def _trace_data(self, value=None, sub_cat='nocat'):
        """
        The method that stores the data in the SQL trace.

        Parameters
        ----------
        value : str or dict
            The value that must be stored in the dictionary.
        sub_cat : str
            The sub category under the function key where the data must be stored.
            the sub_cat becomes the key in the dictionary.

        """

        if self.trace_sql_active:
            if not self.trace_sql_algo in self.trace_sql_log:
                self.trace_sql_log[self.trace_sql_algo] = {}
            if not self.trace_sql_function in self.trace_sql_log[self.trace_sql_algo]:
                self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function] = {}
            if not sub_cat in self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function]:
                self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function][sub_cat] = []

            self.trace_sql_log[self.trace_sql_algo][self.trace_sql_function][sub_cat].append(value)

class DataFrame(object):#pylint: disable=too-many-public-methods
    """
    Represents a frame that is backed by a database SQL statement.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection to the SAP HANA system.
    select_statement : str
        The SQL query backing the dataframe.

        .. note ::
            Parameters beyond ``connection_context`` and ``select_statement`` are intended for internal use. Do not rely on them; they may change without notice.
    """

    # pylint: disable=too-many-public-methods,too-many-instance-attributes
    _df_count = 0

    def __init__(self, connection_context, select_statement, _name=None):
        self.connection_context = connection_context
        self.select_statement = select_statement
        self._columns = None
        self._dtypes = None
        if _name is None:
            self._name = 'DT_{}'.format(DataFrame._df_count)
            self._quoted_name = quotename(self._name)
            DataFrame._df_count += 1
        else:
            self._name = _name
            self._quoted_name = quotename(_name)
        self._ttab_handling = 'unknown'
        self._ttab_reference = None
        self.source_table = None
        self.index = None
        self._validate_columns = True

    @property
    def columns(self):
        """
        Lists the current DataFrame's column names. Computed lazily and cached.
        Each access to this property creates a new copy; mutating the list does not alter or corrupt the DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        list of str
            A list of column names.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.columns
        ['sepal length (cm)',
         'sepal width (cm)',
         'petal length (cm)',
         'petal width (cm)',
         'target']
        """

        if self._columns is None:
            self._columns = self.__populate_columns()
        return self._columns[:]

    @property
    def shape(self):
        """
        Computes the shape of the SAP HANA DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        tuple
            (The number of rows, the number of columns) in the SAP HANA DataFrame.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.shape
            (1, 3)
        """
        return [self.count(), len(self.columns)]

    @property
    def name(self):
        """
        Returns the name of the DataFrame. This value does not correspond to a SAP HANA table name.
        This value is useful for joining predicates when the joining DataFrames have columns with the same name.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A str of DataFrame name.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.name
        'DT_1'
        """

        return self._name

    @property
    def quoted_name(self):
        """
        Specifies the escaped name of the original DataFrame.
        Default-generated DataFrame names are safe to use in SQL without escaping, but names set with DataFrame.alias may require escaping.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A str of DataFrame name.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.quoted_name
        '"DT_1"'
        """

        return self._quoted_name

    def __getitem__(self, index):
        if (isinstance(index, list) and
                all(isinstance(col, _STRING_TYPES) for col in index)):
            return self.select(*index)
        raise TypeError(
            '__getitem__ argument not understood: {!r}'.format(index))

    def __populate_columns(self):
        nodata_select = 'SELECT * FROM ({}) WHERE 0=1'.format(
            self.select_statement)
        with self.connection_context.connection.cursor() as cur:
            cur.execute(nodata_select)
            return [descr[0] for descr in cur.description]

    def __run_query(self, query, fetch_size=32000):
        """
        Runs a query over the DataFrame's connection.

        Parameters
        ----------
        query : str
            The SQL statement to run.
        fetch_size : int
            Specify the fetch size to improve the fetch performance.

            Defaults to 32000.

        Returns
        -------
        list of hdbcli.resultrow.ResultRow objects. A list of query results.
        """
        with self.connection_context.connection.cursor() as cur:
            if not self.connection_context.pyodbc_connection:
                cur.setfetchsize(fetch_size)
            cur.execute(query)
            return cur.fetchall()


    @staticmethod
    def _get_list_of_str(cols, message):
        """
        Generates a list of values after checking that all values in the list are strings. If cols is a string, then a list is created and returned.

        Parameters
        ----------
        cols : str or list of str
            The list of columns to check.
        message : str
            An error message in case the type of the cols parameter is incorrect.

        Returns
        -------
        list
            cols if it is already a list; otherwise, a list containing cols.
        """

        if isinstance(cols, _STRING_TYPES):
            cols = [cols]
        if (not cols or not isinstance(cols, list) or
                not all(isinstance(col, _STRING_TYPES) for col in cols)):
            raise TypeError(message)
        return cols

    def _validate_columns_in_dataframe(self, cols):
        """
        Checks if the specified columns are in the DataFrame.
        Raises an error if any column in cols is not in the current DataFrame.

        Parameters
        ----------
        cols : list
            The list of columns to check.
        """
        if self._validate_columns:
            valid_set = set(self.columns)
            check_names = set(col for col in cols
                            if isinstance(col, _STRING_TYPES))
            if not valid_set.issuperset(check_names):
                invalid_names = [name for name in check_names
                                if name not in valid_set]
                message = "Column(s) not in DataFrame: {}".format(invalid_names)
                raise ValueError(message)

    def declare_lttab_usage(self, usage):
        """
        Declares whether this DataFrame makes use of local temporary tables.

        Some SAP HANA PAL execution routines can execute more efficiently if they know up front whether a DataFrame's SELECT statement requires
        access to local temporary tables.

        Parameters
        ----------
        usage : bool
            Specifies whether this DataFrame uses local temporary tables.
        """

        if self._ttab_handling == 'safe':
            if usage:
                raise ValueError(
                    "declare_lttab_usage(True) called on a DataFrame " +
                    "believed to involve no local temporary tables.")
        elif self._ttab_handling in ('unsafe', 'ttab'):
            if not usage:
                raise ValueError(
                    "declare_lttab_usage(False) called on a DataFrame " +
                    "believed to involve local temporary tables.")
        else:
            self._ttab_handling = 'unsafe' if usage else 'safe'

    def disable_validate_columns(self):
        """
        Disable the column validation.
        """
        self._validate_columns = False

    def enable_validate_columns(self):
        """
        Enable the column validation.
        """
        self._validate_columns = True

    def _propagate_safety(self, other):
        # pylint:disable=protected-access
        if self._ttab_handling == 'safe':
            other._ttab_handling = 'safe'
        elif self._ttab_handling in ('ttab', 'unsafe'):
            other._ttab_handling = 'unsafe'

    def _df(self, select_statement, name=None, propagate_safety=False):
        # Because writing "DataFrame(self.connection_context" everywhere
        # is way too verbose.
        retval = DataFrame(self.connection_context, select_statement, name)
        if propagate_safety:
            self._propagate_safety(retval)
        return retval

    def add_id(self, id_col=None, ref_col=None):
        """
        Return a new SAP HANA DataFrame with a added <id_col> column.

        Parameters
        ----------
        id_col : str, optional
            The name of new ID column.

            Defaults to "ID".
        ref_col : str or list of str, optional
            The id is generated based on ref_col.

        Returns
        -------
        DataFrame
            A new SAP HANA DataFrame with a added <id_col> column.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.collect()
        X    Y
        20   30
        40   5

        >>> df.add_id(id_col='ID')
        ID    X    Y
        1     20   30
        2     40   5
        """
        if id_col is None:
            id_col = "ID"
        order_by = ''
        if ref_col is not None:
            if isinstance(ref_col, (list, tuple)):
                order_by = "ORDER BY {} ASC".format(", ".join(list(map(quotename, ref_col))))
            else:
                order_by = "ORDER BY {} ASC".format(quotename(ref_col))
        select_statement = "SELECT CAST(ROW_NUMBER() OVER({}) AS INTEGER) AS {}, * FROM ({})".format(order_by,
                                                                                                     quotename(id_col),
                                                                                                     self.select_statement)
        return self._df(select_statement, propagate_safety=True)

    def add_constant(self, column_name, value):
        """
        Add a new column with constant value.
        """
        select_statement = "SELECT *, '{}' AS {} FROM ({})".format(value,
                                                                 quotename(column_name),
                                                                 self.select_statement)
        return self._df(select_statement, propagate_safety=True)

    def alias(self, alias):
        """
        Returns a new SAP HANA DataFrame with an alias set.

        Parameters
        ----------
        alias : str
            The name of the DataFrame.

        Returns
        -------
        DataFrame
            A SAP HANA DataFrame with an alias set.

        See Also
        --------
        DataFrame.rename_columns : For renaming individual columns.

        """

        retval = self._df(self.select_statement, alias)
        # pylint:disable=protected-access
        retval._ttab_handling = self._ttab_handling
        retval._ttab_reference = self._ttab_reference
        return retval

    def count(self):
        """
        Computes the number of rows in the SAP HANA DataFrame.

        Parameters
        ----------
        None

        Returns
        -------
        int
            The number of rows in the SAP HANA DataFrame.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.count()
        150
        """

        new_select_statement = "SELECT COUNT(*) FROM ({})".format(self.select_statement)
        try:
            results = self.__run_query(new_select_statement)
        except dbapi.Error as exc:
            logger.error("Failed to get row count for the current Dataframe, %s", exc)
            raise
        except pyodbc.Error as exc:
            logger.error("Failed to get row count for the current Dataframe, %s", str(exc.args[1]))
            raise
        return results[0][0]

    def diff(self, index, periods=1):
        """
        Returns a new SAP HANA DataFrame with differenced values.

        Parameters
        ----------
        index : int or str
            Index of the SAP HANA DataFrame.
        periods : int, optional
            Periods to shift for calculating difference, accepts negative values.

            Defaults to 1.

        Returns
        -------
        DataFrame
            DataFrame with differenced values. No calculation happens if it contains string.
        """
        select_statement = 'SELECT '
        for column in self.columns:
            if column not in index:
                if periods >= 0:
                    select_statement = select_statement + '{col} - LAG({col}, {periods}) OVER(ORDER BY {idx}) {col}, '\
                    .format(col=quotename(column), periods=periods, idx=quotename(index))
                else:
                    select_statement = select_statement + 'LAG({col}, {periods}) OVER(ORDER BY {idx}) - {col} {col}, '\
                    .format(col=quotename(column), periods=-1 * periods, idx=quotename(index))
        select_statement = select_statement[:-2]
        select_statement = select_statement + ' FROM ({})'.format(self.select_statement)
        return self._df(select_statement, propagate_safety=True)

    def drop(self, cols):
        """
        Returns a new SAP HANA DataFrame without the specified columns.

        Parameters
        ----------
        cols : list of str
            The list of column names to be dropped.

        Returns
        -------
        DataFrame
            A new SAP HANA DataFrame that retains only the columns not listed in ``cols``.

        Examples
        --------
        >>> df.collect()
           A  B
        0  1  3
        1  2  4
        >>> df.drop(['B']).collect()
           A
        0  1
        1  2
        """
        if isinstance(cols, str):
            cols = [cols]
        dropped_set = set(cols)
        if not dropped_set.issubset(self.columns):
            own_columns = set(self.columns)
            invalid_columns = [col for col in cols if col not in own_columns]
            raise ValueError("Can't drop nonexistent column(s): {}".format(invalid_columns))

        cols_kept = [quotename(col) for col in self.columns if col not in dropped_set]
        cols_kept = ', '.join(cols_kept)
        # TO DO: if cols_kept are the same as self.columns, then return nothing/self
        select_template = 'SELECT {} FROM ({}) AS {}'
        new_select_statement = select_template.format(
            cols_kept, self.select_statement, self.quoted_name)
        return self._df(new_select_statement, propagate_safety=True)

    def _generate_colname(self, prefix='GEN_COL'):
        # If the input prefix is safe to use unquoted, the output name
        # will be safe to use unquoted too.
        # Otherwise, you'll probably want to quotename the result before
        # using it in SQL.
        if not prefix:
            prefix = 'GEN_COL'
        if prefix not in self.columns:
            return prefix
        for i in range(1+len(self.columns)):
            gen_col = '{}_{}'.format(prefix, i)
            if gen_col not in self.columns:
                return gen_col
        # To get here, we would have to try more new names than this dataframe
        # has columns, and we would have to find that all of those names
        # were taken.
        raise AssertionError("This shouldn't be reachable.")

    def distinct(self, cols=None):
        """
        Returns a new SAP HANA DataFrame with distinct values for the specified columns.
        If no columns are specified, then the distinct row values from all columns are returned.

        Parameters
        ----------
        cols : str or list of str, optional
            A column or list of columns to consider when getting distinct
            values. Defaults to use all columns.

        Returns
        -------
        DataFrame
            The DataFrame with distinct values for cols.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B    C
        0  1  A  100
        1  1  A  101
        2  1  A  102
        3  1  B  100
        4  1  B  101
        5  1  B  102
        6  1  B  103
        7  2  A  100
        8  2  A  100

        Distinct values in a column:

        >>> df.distinct("B").collect()
           B
        0  A
        1  B

        Distinct values of a subset of columns:

        >>> df.distinct(["A", "B"]).collect()
           A  B
        0  1  B
        1  2  A
        2  1  A

        Distinct values of the entire data set:

        >>> df.distinct().collect()
           A  B    C
        0  1  A  102
        1  1  B  103
        2  1  A  101
        3  2  A  100
        4  1  B  101
        5  1  A  100
        6  1  B  100
        7  1  B  102
        """

        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns
        select_statement = "SELECT DISTINCT {} FROM ({}) AS {}".format(
            ', '.join([quotename(col) for col in cols]),
            self.select_statement, self.quoted_name)
        return self._df(select_statement, propagate_safety=True)

    def drop_duplicates(self, subset=None):
        """
        Returns a new SAP HANA DataFrame with duplicate rows removed. All columns in the
        DataFrame are returned. There is no way to keep specific duplicate rows.

        .. warning::
           Specifying a non-None value of ``subset`` may produce an unstable \
           DataFrame, the contents of which may be different every time you \
           look at it. Specifically, if two rows are duplicates in their \
           ``subset`` columns and have different values in other columns, \
           Then a different row could be picked every time you look at the result.

        Parameters
        ----------
        subset : list of str, optional
            A list of columns to consider when deciding whether rows are \
            duplicates of each other. Defaults to use all columns.

        Returns
        -------
        DataFrame
            A DataFrame with only one copy of duplicate rows.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B    C
        0  1  A  100
        1  1  A  101
        2  1  A  102
        3  1  B  100
        4  1  B  101
        5  1  B  102
        6  1  B  103
        7  2  A  100
        8  2  A  100

        Drop duplicates based on the values of a subset of columns:

        >>> df.drop_duplicates(["A", "B"]).collect()
           A  B    C
        0  1  A  100
        1  1  B  100
        2  2  A  100

        Distinct values on the entire data set:

        >>> df.drop_duplicates().collect()
           A  B    C
        0  1  A  102
        1  1  B  103
        2  1  A  101
        3  2  A  100
        4  1  B  101
        5  1  A  100
        6  1  B  100
        7  1  B  102
        """

        if subset is None:
            return self._df("SELECT DISTINCT * FROM ({}) AS {}".format(
                self.select_statement, self.quoted_name), propagate_safety=True)

        if not subset:
            raise ValueError("drop_duplicates requires at least one column in subset")

        keep_columns = ', '.join([quotename(col) for col in self.columns])
        partition_by = ', '.join([quotename(col) for col in subset])
        seqnum_col = quotename(self._generate_colname('SEQNUM'))
        seqnum_template = "SELECT *, ROW_NUMBER() OVER (PARTITION BY {}) AS {} FROM ({})"
        select_with_seqnum = seqnum_template.format(
            partition_by, seqnum_col, self.select_statement)
        new_select_statement = "SELECT {} FROM ({}) WHERE {} = 1".format(
            keep_columns, select_with_seqnum, seqnum_col)
        return self._df(new_select_statement, propagate_safety=True)

    def dropna(self, how=None, thresh=None, subset=None):
        # need to test
        """
        Returns a new DataFrame with NULLs removed.

        Parameters
        ----------
        how : {'any', 'all'}, optional
            If provided, 'any' eliminates rows with any NULLs, \
            and 'all' eliminates rows that are entirely NULLs. \
            If neither ``how`` nor ``thresh`` are provided, ``how`` \
            defaults to 'any'.
        thresh : int, optional
            If provided, rows with fewer than ``thresh`` non-NULL values \
            are dropped. \
            You cannot specify both ``how`` and ``thresh``.
        subset : list of str, optional
            The columns to consider when looking for NULLs. Values in
            other columns are ignored, whether they are NULL or not.
            Defaults to all columns.

        Returns
        -------
        DataFrame
            A new SAP HANA DataFrame with a SELECT statement that removes NULLs.

        Examples
        --------
        Dropping rows with any NULL:

        >>> df.collect()
             A    B    C
        0  1.0  3.0  5.0
        1  2.0  4.0  NaN
        2  3.0  NaN  NaN
        3  NaN  NaN  NaN
        >>> df.dropna().collect()
             A    B    C
        0  1.0  3.0  5.0

        Dropping rows that are entirely nulls:

        >>> df.dropna(how='all').collect()
             A    B    C
        0  1.0  3.0  5.0
        1  2.0  4.0  NaN
        2  3.0  NaN  NaN

        Dropping rows with less than 2 non-null values:

        >>> df.dropna(thresh=2).collect()
             A    B    C
        0  1.0  3.0  5.0
        1  2.0  4.0  NaN
        """

        if how is not None and thresh is not None:
            raise ValueError("Cannot provide both how and thresh.")

        if subset is not None:
            cols = subset
        else:
            cols = self.columns

        cols = [quotename(col) for col in cols]
        if thresh is None:
            if how in {'any', None}:
                and_or = ' OR '
            elif how == 'all':
                and_or = ' AND '
            else:
                raise ValueError("Invalid value of how: {}".format(how))
            drop_if = and_or.join('{} IS NULL'.format(col) for col in cols)
            keep_if = 'NOT ({})'.format(drop_if)
            retval = self.filter(keep_if)
            self._propagate_safety(retval)
            return retval

        count_expression = '+'.join(
            ['(CASE WHEN {} IS NULL THEN 0 ELSE 1 END)'.format(col) for col in cols])
        count_colname = self._generate_colname('CT')
        select_with_count = 'SELECT *, ({}) AS {} FROM ({}) {}'.format(
            count_expression, count_colname, self.select_statement, self.quoted_name)
        projection = ', '.join([quotename(col) for col in self.columns])
        new_select_statement = 'SELECT {} FROM ({}) WHERE {} >= {}'.format(
            projection, select_with_count, count_colname, thresh)
        return self._df(new_select_statement, propagate_safety=True)

    def deselect(self, cols):
        """
        Returns a new DataFrame without columns derived from the current DataFrame.

        Parameters
        ----------
        cols : str or tuple/list of str.
            The columns are excluded in the new DataFrame.

        Returns
        -------
        DataFrame
            A new DataFrame object excluding the specified columns.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B  C
        0  1  2  3

        Selecting a subset of existing columns:

        >>> df.deselect(['A', 'B']).collect()
           C
        0  3
        """
        columns = []
        if isinstance(cols, str):
            cols = [cols]
        for col in self.columns:
            if col not in cols:
                columns.append(col)
        return self.select(*columns)

    def dtypes(self, subset=None):
        """
        Returns a sequence of tuples describing the DataFrame's SQL types.

        The tuples list the name, SQL type name, display_size, internal_size,
        precision and scale corresponding to the DataFrame's columns.

        Parameters
        ----------
        subset : list of str, optional
            The columns that the information is generated from.
            Defaults to all columns.

        Returns
        -------
        dtypes : list of tuples
            Each tuple consists of the name, SQL type name, display_size, internal_size,
            precision and scale for one of the DataFrame's columns. The list is in the order
            specified by the ``subset``, or in the DataFrame's original column
            order if a ``subset`` is not provided.
        """

        if self._dtypes is None:
            with self.connection_context.connection.cursor() as cur:
                cur.execute(self.select_statement)
                if self.connection_context.pyodbc_connection:
                    self._dtypes = [(c[0],
                                     "VARCHAR" if c[1].__name__.upper() == "STR" else c[1].__name__.upper(),
                                     c[3],
                                     c[3],
                                     c[4],
                                     c[5]) for c in cur.description]
                else:
                    self._dtypes = [(c[0], TYPE_CODES[c[1]], c[2], c[3], c[4], c[5]) for c in cur.description]
        if subset is None:
            return self._dtypes[:]
        dtype_map = {descr[0]: descr for descr in self._dtypes}
        return [dtype_map[col] for col in subset]

    _ARBITRARY_PSEUDOTOKEN_LIMIT = 200

    def empty(self):
        """
        Returns True if this DataFrame has 0 rows.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            True if the DataFrame is empty.

        Notes
        -----
        If a DataFrame contains only NULLs, it is not considered empty.

        Examples
        --------
        >>> df1.collect()
        Empty DataFrame
        Columns: [ACTUAL, PREDICT]
        Index: []
        >>> df1.empty()
        True

        >>> df2.collect()
          ACTUAL PREDICT
        0   None    None
        >>> df2.empty()
        False
        """
        return self.count() == 0

    def _token_validate(self, sql):
        """
        Calls IS_SQL_INJECTION_SAFE on input. Does not guarantee injection safety.

        Parameters
        ----------
        sql : str
            A SQL statement.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If IS_SQL_INJECTION_SAFE returns 0.

        Notes
        -----
        This method does not guarantee injection safety. Any parts of this library
        that take SQL statements, expressions, or predicates are not safe
        against SQL injection. This method only catches some instances of comments
        and malformed tokens in the input.

        IS_SQL_INJECTION_SAFE may produce false positives or false negatives.
        """

        with self.connection_context.connection.cursor() as cur:
            cur.execute('SELECT IS_SQL_INJECTION_SAFE(?, ?) FROM DUMMY',
                        (sql, self._ARBITRARY_PSEUDOTOKEN_LIMIT))
            val = cur.fetchone()[0]
        if not val:
            msg = 'SQL token validation failed for string {!r}'.format(sql)
            raise BadSQLError(msg)

    def filter(self, condition):
        """
        Selects rows that match the given condition.

        Very little checking is done on the condition string.
        Use only with trusted inputs.

        Parameters
        ----------
        condition : str
            A filter condition. Format as SQL <condition>.

        Returns
        -------
        DataFrame
            A DataFrame with rows that match the given condition.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If comments or malformed tokens are detected in ``condition``.
            May have false positives and false negatives.

        Examples
        --------
        >>> df.collect()
           A  B
        0  1  3
        1  2  4
        >>> df.filter('B < 4').collect()
           A  B
        0  1  3
        """

        return self._df('SELECT * FROM ({}) AS {} WHERE {}'.format(
            self.select_statement, self.quoted_name, condition))

    def has(self, col):
        """
        Returns True if a column is in the DataFrame.

        Parameters
        ----------
        col : str
            The name of column to search in the projection list of this DataFrame.

        Returns
        -------
        bool
            Returns True if the column exists in the DataFrame's projection list.

        Examples
        --------
        >>> df.columns
        ['A', 'B']
        >>> df.has('A')
        True
        >>> df.has('C')
        False
        """

        # df.has(col) doesn't really give much benefit over using
        # col in df.columns directly. It may not be worth
        # having this method at all.
        return col in self.columns

    def head(self, n=1): #pylint: disable=invalid-name
        """
        Returns a DataFrame of the first ``n`` rows in the current DataFrame.

        Parameters
        ----------
        n : int, optional
            The number of rows returned.

            Defaults to 1.

        Returns
        -------
        DataFrame
            A new DataFrame of the first ``n`` rows of the current DataFrame.

        """

        head_select = 'SELECT TOP {} * FROM ({}) dt'.format(
            n, self.select_statement)
        return self._df(head_select, propagate_safety=True)

    def hasna(self, cols=None):
        """
        Returns True if a DataFrame contains NULLs.

        Parameters
        ----------
        cols : str or list of str, optional
            A column or list of columns to be checked for NULL values.
            Defaults to all columns.

        Returns
        -------
        bool
            True if this DataFrame contains NULLs.

        Examples
        --------

        >>> df1.collect()
          ACTUAL PREDICT
        0   1.0    None

        >>> df1.hasna()
        True
        """

        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns

        count_cols = []
        for col in cols:
            quoted_col = quotename(col)
            count_cols.append("count({})".format(quoted_col))
        minus_expression = ' + '.join(count_cols)

        count_statement = "SELECT COUNT(*)*{} - ({}) FROM ({})".format(
            len(cols),
            minus_expression,
            self.select_statement)
        try:
            count = self.__run_query(count_statement)
        except dbapi.Error as exc:
            logger.error("Failed to get NULL value count for the current Dataframe, %s", exc)
            raise
        except pyodbc.Error as exc:
            logger.error("Failed to get NULL value count for the current Dataframe, %s", str(exc.args[1]))
            raise
        return count[0][0] > 0

    def fillna(self, value, subset=None):
        """
        Returns a DataFrame with NULLs replaced with a specified value.

        Parameters
        ----------
        value : int or float
            The value that replaces NULL. ``value`` should have a type that is
            appropriate for the selected columns.
        subset : list of str, optional
            A list of columns whose NULL values will be replaced.
            Defaults to all columns.

        Returns
        -------
        DataFrame
            A new DataFrame with the NULL values replaced.
        """

        if subset is not None:
            filled_set = set(subset)
        else:
            filled_set = set(self.columns)
        if not isinstance(value, (float, str) + _INTEGER_TYPES):
            raise TypeError("Fill values currently must be ints, str or floats.")
        #untouched columns
        select_values = []
        tab_struct = self.get_table_structure()
        for col in self.columns:
            quoted_col = quotename(col)
            if col in filled_set:
                #pylint: disable=W0511
                if isinstance(value, str) and ('VARCHAR' in tab_struct[col].upper() or
                                               'TIME' in tab_struct[col].upper() or
                                               'DATE' in tab_struct[col].upper() or
                                               'TEXT' in tab_struct[col].upper()):
                    select_values.append("COALESCE({0}, '{1}') AS {0}".format(quoted_col, value))
                elif isinstance(value, (int, float)) and ('INT' in tab_struct[col].upper() or
                                                          'DOUBLE' in tab_struct[col].upper() or
                                                          'DECIMAL' in tab_struct[col].upper()):
                    select_values.append("COALESCE({0}, {1}) AS {0}".format(quoted_col, value))
                else:
                    select_values.append(quoted_col)
            else:
                select_values.append(quoted_col)
        cols = ', '.join(select_values)
        new_select_statement = 'SELECT {} FROM ({}) dt'.format(cols, self.select_statement)
        return self._df(new_select_statement, propagate_safety=True)

    def get_table_structure(self):
        """
        Return dict format table structure.
        """
        table_structure = {}
        for item in self.dtypes():
            if 'VARCHAR' in item[1].upper():
                table_structure[item[0]] = "{}({})".format(item[1], item[4])
            elif item[1].upper() == 'DECIMAL':
                table_structure[item[0]] = "{}({}, {})".format(item[1], item[4], item[5])
            else:
                table_structure[item[0]] = item[1]
        return table_structure

    def join(self, other, condition=None, how='inner', select=None):
        """
        Returns a new DataFrame that is a join of the current DataFrame with
        another specified DataFrame.

        Parameters
        ----------
        other : DataFrame or list of DataFrame
            The DataFrame to join with.

        condition : str, optional
            The join predicate. If index has been set, use the index as key to join.

            Defaults to None.

        how : {'inner', 'left', 'right', 'outer', 'cross'}, optional
            The type of join. Defaults to 'inner'.

            Defaults to 'inner'.

        select : list, optional
            If provided, each element specifies a column in the result.
            A string in the ``select`` list should be the name of a column in
            one of the input DataFrames. A (expression, name) tuple creates
            a new column with the given name, computed from the given
            expression.

            If this value is not provided, defaults to selecting all columns from both
            DataFrames, with the left DataFrame's columns first.

        Returns
        -------
        DataFrame
            A new DataFrame object made from the join() of the current DataFrame
            with another DataFrame.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If comments or malformed tokens are detected in ``condition``
            or in a column expression.
            May have false positives and false negatives.

        Examples
        --------
        Use the expression selection functionality to disambiguate duplicate
        column names in a join():

        >>> df1.collect()
           A  B    C
        0  1  2  3.5
        1  2  4  5.6
        2  3  3  1.1

        >>> df2.collect()
           A  B     D
        0  2  1  14.0
        1  3  4   5.6
        2  4  3   0.0

        Old method:

        >>> df1.alias('L').join(df2.alias('R'), 'L.A = R.A', select=[
        ...     ('L.A', 'A'),
        ...     ('L.B', 'B_LEFT'),
        ...     ('R.B', 'B_RIGHT'),
        ...     'C',
        ...     'D']).collect()
           A  B_LEFT  B_RIGHT    C     D
        0  2       4        1  5.6  14.0
        1  3       3        4  1.1   5.6

        New method:

        >>> df1.set_index("A").join(df2.rename_columns({"B":"B2"}).set_index("A")).collect()
           A  B B2    C   D
        0  2  4  1  5.6  14
        1  3  3  4  1.1 5.6
        """

        #if left and right joins are improper (ex: int on string) then there will be a sql error
        join_type_map = {
            'inner': 'INNER',
            'left': 'LEFT OUTER',
            'right': 'RIGHT OUTER',
            'outer': 'FULL OUTER',
            'cross': 'CROSS'
        }
        join_type = join_type_map[how]
        if condition is not None:
            if how not in ['inner', 'outer', 'left', 'right']:
                raise ValueError('Invalid value for "how" argument: {!r}'.format(how))
            self._token_validate(condition)
            on_clause = 'ON ' + condition


            if select is None:
                projection_string = '*'
            else:
                projection_string = self._stringify_select_list(select)

            select_template = 'SELECT {} FROM ({}) AS {} {} JOIN ({}) AS {} {}'
            new_select_statement = select_template.format(
                projection_string,
                self.select_statement, self.quoted_name, join_type,
                other.select_statement, other.quoted_name, on_clause)
        else:
            if not isinstance(other, (list, tuple)):
                other = [other]
            if self.index is None:
                raise ValueError("Index has not been set!")
            sel_part1 = "SELECT " + ", ".join(['T0.' + quotename(col) for col in self.columns]) + ", "
            sel_part2 = "\nFROM ({}) T0 ".format(self.select_statement)
            for idx, hana_df in enumerate(other):
                if hana_df.index is None:
                    raise ValueError("Index has not been set!")
                if isinstance(self.index, (list, tuple)):
                    sel_part1 = sel_part1 + ", ".join(["T{}.".format(idx + 1) + quotename(col) for col in hana_df.columns if col not in hana_df.index])
                    conditions = []
                    if len(self.index) != len(hana_df.index):
                        raise ValueError("Index list lenth does not match!")
                    for idx_item, other_idex_item in zip(self.index, hana_df.index):
                        conditions.append("T0.{0} = T{1}.{2}".format(quotename(idx_item), idx + 1, quotename(other_idex_item)))
                    sel_part2 = sel_part2 + "{0} JOIN ({1}) T{2}\n ON {3}\n".format(join_type,
                                                                                    hana_df.select_statement,
                                                                                    idx + 1,
                                                                                    " AND ".join(conditions))
                else:
                    sel_part1 = sel_part1 + ", ".join(["T{}.".format(idx + 1) + quotename(col) for col in hana_df.columns if col != hana_df.index])
                    sel_part2 = sel_part2 + "{0} JOIN ({1}) T{2}\n ON T0.{3} = T{2}.{4}\n".format(join_type,
                                                                                                  hana_df.select_statement,
                                                                                                  idx + 1,
                                                                                                  quotename(self.index),
                                                                                                  quotename(hana_df.index))
                if idx < len(other) - 1:
                    sel_part1 = sel_part1 + ", "
            new_select_statement = sel_part1 + sel_part2
        return self._df(new_select_statement)

    def set_name(self, name):
        """
        Set the name of the DataFrame.

        Parameters
        ----------
        name : str
            The name of dataframe.
        """
        self._name = name

    def set_index(self, keys):
        """
        Set the DataFrame using the existing columns.

        Parameters
        ----------
        keys : str or list of str
            This parameter can be either a single column key or a list of column keys.
        """
        if isinstance(keys, str):
            if keys not in self.columns:
                raise ValueError("keys not in columns!")
        else:
            for key in keys:
                if key not in self.columns:
                    raise ValueError("keys not in columns!")
        self.index = keys
        return self

    def save(self, where, table_type=None, force=False, save_source=True, append=False, data_lake=False, data_lake_container='SYSRDL#CG', view_structure=None):
        """
        Creates a table or view holding the current DataFrame's data.

        Parameters
        ----------
        where : str or (str, str) tuple
            The table name or (schema name, table name) tuple. If no schema
            is provided, then the table or view is created in the current
            schema.
        table_type : str, optional
            The type of table to create. The value is case insensitive.

            Permanent table options:
              - "ROW"
              - "COLUMN"
              - "HISTORY COLUMN"

            Temporary table options:
              - "GLOBAL TEMPORARY"
              - "GLOBAL TEMPORARY COLUMN"
              - "LOCAL TEMPORARY"
              - "LOCAL TEMPORARY COLUMN"

            Not a table:
              - "VIEW"

            Defaults to 'LOCAL TEMPORARY COLUMN' if ``where`` starts
            with '#'. Otherwise, the default is 'COLUMN'.
        force : bool, optional
            If force is True, it will replace the existing table.
        save_source : bool, optional
            If True, it will save the name of source table.

            Defaults to True.
        append : bool, optional
            If True, it will use the existing table and append data to it.

            Defaults to False.
        data_lake : bool, optional
            If True, it will save the table to HANA data lake.

            Defaults to False.
        view_structure : dict, optional
            Only valid when `table_type="VIEW"`. Define the parameters in the view.

            Defaults to None.

        Returns
        -------
        DataFrame
            A DataFrame that represents the new table or view.

        Notes
        -----
        For this operation to succeed, the table name must not be in
        use, the schema must exist, and the user must have permission
        to create tables (or views) in the target schema.
        """

        if isinstance(where, tuple):
            schema, table = where
        else:
            schema, table = None, where
        if view_structure:
            if table_type != "VIEW":
                raise ValueError('table_type must be "VIEW".')
        if data_lake:
            if force:
                self.connection_context.drop_table(table=table, data_lake=True, data_lake_container=data_lake_container) #drop data lake table
                self.connection_context.drop_table(table=table, schema=schema) #drop virtual table
            self.connection_context.copy_to_data_lake(data=self,
                                                      virtual_table=table,
                                                      data_lake_table=table,
                                                      schema=schema,
                                                      append=append,
                                                      data_lake_container=data_lake_container)
        else:
            if table_type is None:
                if table.startswith('#'):
                    table_type = 'LOCAL TEMPORARY COLUMN'
                else:
                    table_type = 'COLUMN'
            if table_type.upper() not in {
                    'ROW',
                    'COLUMN',
                    'HISTORY COLUMN',
                    'GLOBAL TEMPORARY',
                    'GLOBAL TEMPORARY COLUMN',
                    'LOCAL TEMPORARY',
                    'LOCAL TEMPORARY COLUMN',
                    'VIEW'}:
                raise ValueError("{!r} is not a valid value of table_type".format(
                    table_type))

            has_table = False
            try:
                has_table = self.connection_context.has_table(table, schema)
            except dbapi.Error as err:
                logger.warning(err)
                pass
            except pyodbc.Error as err:
                logger.warning(err)
                pass
            if schema is None:
                where_string = quotename(table)
            else:
                where_string = '{}.{}'.format(*map(quotename, where))

            table_type = table_type.upper()
            not_created = True
            if table_type != 'VIEW':
                table_type += ' TABLE'
            with self.connection_context.connection.cursor() as cur:
                if (force is True) and (table_type != 'VIEW') and (append is False):
                    if has_table:
                        try:
                            execute_logged(cur,
                                           "DROP TABLE {};".format(where_string),
                                           self.connection_context.sql_tracer,
                                           self.connection_context)
                            has_table = False
                        except dbapi.Error:
                            pass
                        except pyodbc.Error:
                            pass
                if (force is True) and (table_type == 'VIEW') and (append is False):
                    try:
                        execute_logged(cur,
                                       "DROP VIEW {};".format(where_string),
                                       self.connection_context.sql_tracer,
                                       self.connection_context)
                    except dbapi.Error:
                        pass
                    except pyodbc.Error:
                        pass
                if not has_table:
                    if view_structure:
                        view_tt = []
                        for param_key, param_value in view_structure.items():
                            view_tt.append("IN {} {}".format(param_key, param_value))
                        execute_logged(cur,
                                    "CREATE {} {} ({}) AS ({})".format(table_type,
                                                                       where_string,
                                                                       ", ".join(view_tt),
                                                                       self.select_statement),
                                    self.connection_context.sql_tracer,
                                    self.connection_context)
                    else:
                        execute_logged(cur,
                                       "CREATE {} {} AS ({})".format(table_type,
                                                                     where_string,
                                                                     self.select_statement),
                                       self.connection_context.sql_tracer,
                                       self.connection_context)
                    not_created = False
                if append and not_created:
                    execute_logged(cur,
                                   "INSERT INTO {} {}".format(where_string, self.select_statement),
                                   self.connection_context.sql_tracer,
                                   self.connection_context)
            if not self.connection_context.pyodbc_connection:
                if not self.connection_context.connection.getautocommit():
                    self.connection_context.connection.commit()
        return self.connection_context.table(table, schema=schema, save_source=save_source)

    def save_nativedisktable(self, where, force=False, save_source=True):
        """
        Materialize dataframe to a SAP HANA native disk.

        Parameters
        ----------
        where : str or (str, str) tuple
            The table name or (schema name, table name) tuple. If no schema
            is provided, then the table or view is created in the current
            schema.
        force : bool, optional
            If force is True, it will replace the existing table.
        save_source : bool, optional
            If True, it will save the name of source table.
            Defaults to True.

        Returns
        -------
        DataFrame
            A DataFrame that represents the new table.

        """
        if isinstance(where, tuple):
            schema, table = where
        else:
            schema, table = None, where
        if schema is None:
            where_string = quotename(table)
        else:
            where_string = '{}.{}'.format(*map(quotename, where))
        with self.connection_context.connection.cursor() as cur:
            if force is True:
                try:
                    execute_logged(cur,
                                   "DROP TABLE {};".format(where_string),
                                   self.connection_context.sql_tracer,
                                   self.connection_context)
                except dbapi.Error:
                    pass
                except pyodbc.Error:
                    pass
            execute_logged(cur,
                           "CREATE COLUMN TABLE {} AS ({})".format(where_string, self.select_statement),
                           self.connection_context.sql_tracer,
                           self.connection_context)
            execute_logged(cur,
                           "ALTER TABLE {} PAGE LOADABLE CASCADE".format(where_string),
                           self.connection_context.sql_tracer,
                           self.connection_context)
        if not self.connection_context.pyodbc_connection:
            if not self.connection_context.connection.getautocommit():
                self.connection_context.connection.commit()
        return self.connection_context.table(table, schema=schema, save_source=save_source)

    def split_column(self, column, separator, new_column_names):
        """
        Returns a new DataFrame with splitted column.

        Parameters
        ----------
        column : str
            A column or list of columns to be splitted.
        separator : str
            The separator.
        new_column_names : list of str
            The splitted column names

        Returns
        -------
        DataFrame
            New DataFrame object with splitted columns as specified.
        """
        substr_list = []
        temp_substr = "SUBSTR_AFTER ({}, '{}') ".format(quotename(column), separator)
        count = 0
        for col in new_column_names:
            if count == 0:
                col_str = "SUBSTR_BEFORE ({}, '{}') {}".format(quotename(column), separator, quotename(col))
            elif count == len(new_column_names) - 1:
                col_str = temp_substr + "{}".format(quotename(col))
            else:
                col_str = "SUBSTR_BEFORE ({}, '{}') {}".format(temp_substr, separator, quotename(col))
                temp_substr = "SUBSTR_AFTER ({}, '{}') ".format(temp_substr, separator)
            count = count + 1
            substr_list.append(col_str)
        new_select_statement = "SELECT *, {} FROM ({})".format(", ".join(substr_list), self.select_statement)
        return self._df(new_select_statement, propagate_safety=True).deselect(column)

    def concat_columns(self, columns, separator):
        """
        Returns a new DataFrame with splitted column.

        Parameters
        ----------
        columns : list of str
            A list of columns to be concatenated.
        separator : str
            The separator.

        Returns
        -------
        DataFrame
            New DataFrame object with concat column as specified.
        """
        concat_str = " || '{}' || ".format(separator)
        new_select_statement = 'SELECT *, {} "{}" FROM ({})'.format(concat_str.join([quotename(col) for col in columns]),
                                                                    separator.join(columns),
                                                                    self.select_statement)
        return self._df(new_select_statement, propagate_safety=True).deselect(columns)

    def sort(self, cols, desc=False):
        """
        Returns a new DataFrame sorted by the specified columns.

        Parameters
        ----------
        cols : str or list of str
            A column or list of columns to sort by.
            If a list is specified, then the sort order in parameter desc is used
            for all columns.
        desc : bool, optional
            Set to True to sort in descending order. Defaults to False,
            for ascending order. Default value is False.

        Returns
        -------
        DataFrame
            New DataFrame object with rows sorted as specified.
        """

        # Issue: DataFrames constructed from a sorted DataFrame may not
        # respect its order, since there's no guarantee that subqueries
        # and derived tables will preserve order.
        #
        # collect() is the only thing we can really guarantee will work.
        #
        # We'd have to change our model to propagate ordering constraints
        # explicitly, outside of the select_statement, to guarantee
        # ordering.
        if not cols:
            raise ValueError("Can't sort by 0 columns")
        cols = DataFrame._get_list_of_str(cols,
                                          'Parameter cols must be a string or a list of strings')
        self._validate_columns_in_dataframe(cols)

        cols = [quotename(c) for c in cols]
        template = '{} DESC' if desc else '{} ASC'
        order_by = 'ORDER BY ' + ', '.join(template.format(col) for col in cols)
        new_select_statement = 'SELECT * FROM ({}) AS {} {}'.format(
            self.select_statement, self.quoted_name, order_by)
        return self._df(new_select_statement, propagate_safety=True)

    def sort_values(self, by, ascending=True):
        """
        Returns a new DataFrame sorted by the specified columns.

        Parameters
        ----------
        by : str or list of str
            A column or list of columns to sort by.
            If a list is specified, then the sort order in parameter desc is used
            for all columns.

        ascending : bool, optional
            Set to False to sort in descending order. Defaults to False,
            for ascending order.

            Defaults to True

        Returns
        -------
        DataFrame
            New DataFrame object with rows sorted as specified.
        """
        return self.sort(cols=by, desc=not ascending)

    def sort_index(self, ascending=True):
        """
        Returns a new DataFrame sorted by the index.

        Parameters
        ----------

        ascending : bool, optional
            Set to False to sort in descending order. Defaults to False,
            for ascending order.

            Defaults to True

        Returns
        -------
        DataFrame
            New DataFrame object with rows sorted as specified.
        """
        if self.index is None:
            raise ValueError("Index has not been set!")
        return self.sort(cols=self.index, desc=not ascending)

    def _stringify_select_list(self, select):
        projection = []
        for col in select:
            if isinstance(col, _STRING_TYPES):
                if '*' in col:
                    projection.append(col)
                else:
                    projection.append(quotename(col))
            else:
                expr, name = col
                self._token_validate(expr)
                projection.append('{} AS {}'.format(expr, quotename(name)))
        return ', '.join(projection)

    def select(self, *cols):
        """
        Returns a new DataFrame with columns derived from the current DataFrame.

        .. warning::
            There is no check that inputs interpreted as SQL expressions are
            actually valid expressions; an "expression" like
            "A FROM TAB; DROP TABLE IMPORTANT_THINGS; SELECT A" can cause
            a lot of damage.

        Parameters
        ----------
        cols : str or (str, str) tuple.
            The columns in the new DataFrame. A string is treated as the name
            of a column to select; a (str, str) tuple is treated as
            (SQL expression, alias). As a special case, '*' is expanded
            to all columns of the original DataFrame.

        Returns
        -------
        DataFrame
            A new DataFrame object with the specified columns.

        Raises
        ------
        hana_ml.ml_exceptions.BadSQLError
            If comments or malformed tokens are detected in a column
            expression. May have false positives and false negatives.

        Examples
        --------
        Input:

        >>> df.collect()
           A  B  C
        0  1  2  3

        Selecting a subset of existing columns:

        >>> df.select('A', 'B').collect()
           A  B
        0  1  2

        Computing a new column based on existing columns:

        >>> df.select('*', ('B * 4', 'D')).collect()
           A  B  C  D
        0  1  2  3  8
        """

        columns = []
        for col in cols:
            if isinstance(col, list):
                # Compatibility with old df.select(['a', 'b']) style.
                columns.extend(col)
            elif col == '*':
                columns.extend(self.columns)
            else:
                columns.append(col)

        self._validate_columns_in_dataframe(columns)

        projection_string = self._stringify_select_list(columns)
        new_select_statement = 'SELECT {} FROM ({}) AS {}'.format(
            projection_string, self.select_statement, self.quoted_name)

        newdf = self._df(new_select_statement)
        newdf._columns = [col if isinstance(col, _STRING_TYPES) else col[1]  #pylint: disable=protected-access
                          for col in columns]
        if all(isinstance(col, _STRING_TYPES) for col in columns):
            self._propagate_safety(newdf)
        return newdf

    def union(self,
              other,
              all=True, # pylint:disable=redefined-builtin
              # by='position',
             ):
        """
        Combines this DataFrame's rows and another DataFrame's rows into
        one DataFrame. This operation is equivalent to a SQL UNION ALL.

        Parameters
        ----------
        other : DataFrame, list of DataFrame
            The right side of the union.
        all : bool, optional
            If True, keep duplicate rows; equivalent to UNION ALL in SQL.
            If False, keep only one copy of duplicate rows (even if they
            come from the same side of the union); equivalent to a UNION
            or a UNION ALL followed by DISTINCT in SQL.
            Defaults to True.

        Returns
        -------
        DataFrame
            The combined data from ``self`` and ``other``.

        Examples
        --------
        We have two DataFrames we want to union, with some duplicate rows:

        >>> df1.collect()
           A  B
        0  1  2
        1  1  2
        2  2  3

        >>> df2.collect()
           A  B
        0  2  3
        1  3  4

        union() produces a DataFrame that contains all rows of both df1
        and df2, like a UNION ALL:

        >>> df1.union(df2).collect()
           A  B
        0  1  2
        1  1  2
        2  2  3
        3  2  3
        4  3  4

        To get the deduplication effect of a UNION DISTINCT, pass
        all=False or call distinct() after union():

        >>> df1.union(df2, all=False).collect()
           A  B
        0  1  2
        1  2  3
        2  3  4
        >>> df1.union(df2).distinct().collect()
           A  B
        0  1  2
        1  2  3
        2  3  4
        """
        if isinstance(other, (list, tuple)):
            new_select_list = ['({})'.format(self.select_statement)]
            for other_df in other:
                new_select_list.append('({})'.format(other_df.select_statement))
            new_select = ' UNION ALL '.join(new_select_list) if all else ' UNION '.join(new_select_list)
        else:
            new_select = '(({}) {} ({}))'.format(
                self.select_statement,
                'UNION ALL' if all else 'UNION',
                other.select_statement)
        retval = self._df(new_select)
        if isinstance(other, (list, tuple)):
            for other_df in other:
                if self._ttab_handling == other_df._ttab_handling == 'safe':
                    retval._ttab_handling = 'safe'
                elif {self._ttab_handling, other_df._ttab_handling} & {'ttab', 'unsafe'}:
                    retval._ttab_handling = 'unsafe'
        else:
            if self._ttab_handling == other._ttab_handling == 'safe':
                retval._ttab_handling = 'safe'
            elif {self._ttab_handling, other._ttab_handling} & {'ttab', 'unsafe'}:
                retval._ttab_handling = 'unsafe'
        return retval

    def collect(self, fetch_size=32000, geometries=True, convert_geo_to_shapely=True):
        """
        Copies the current DataFrame to a new Pandas DataFrame.

        Parameters
        ----------
        fetch_size : int, optional
            Fetch size in hdbcli.
        geometries : bool, optional
            With this flag set to `True` (default),
            the geometries are converted to Well-Known-Text representations
            in the resulting `Pandas` dataframe. Even if they are converted
            to Shapely objects (see `convert_geo_to_shapely`), when you
            print the dataframe, the geometry columns are represented as
            Well-Known-Text.

            If you need the raw binary values, set this flag to `False`
        convert_geo_to_shapely : bool, optional
            If set to `True` (default), all geometry columns will be
            converted to a `Shapely` object, so that the dataframe can
            be directly used in visualization libraries for example.
            If your processing does not support `Shapely` objects, you
            can switch this conversion off. In this case the columns
            reamin of type `String`.

            **Note:** Before the conversion, it is checked, if any value
            in the column is `Null`. If so, this column will **not** be
            converted to a `Shapely` object, because `Null` WKT strings
            are not supported by `Shapely`.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame that contains the current DataFrame's data.

        Examples
        --------
        Viewing a hana_ml DataFrame doesn't execute the underlying SQL or fetch the data:

        >>> df = cc.table('T')
        >>> df
        <hana_ml.dataframe.DataFrame object at 0x7f2b7f24ddd8>

        Using collect() executes the SQL and fetchs the results into a Pandas DataFrame:

        >>> df.collect()
           A  B
        0  1  3
        1  2  4

        >>> type(df.collect())
        <class 'pandas.core.frame.DataFrame'>
        """
        select_statement = self.select_statement
        if not self.connection_context.pyodbc_connection:
            try:
                if geometries:
                    geo_cols = self.geometries
                    if len(geo_cols) == 0:
                        select_statement = self.select_statement
                    else:
                        sql_cols = ""

                        for col in self.columns:
                            if col in geo_cols:
                                sql_cols = ", ".join((sql_cols, '"{}".ST_AsWKT()'.format(col)))
                            else:
                                sql_cols = ", ".join((sql_cols, '"{}"'.format(col)))

                        sql_cols = sql_cols[2:]  # Clean up leading ", "

                        select_statement = "SELECT {cols} from ({source})".format(
                            cols=sql_cols, source=self.select_statement
                        )
            except dbapi.Error as exc:
                logger.warning("Errors in checking geometries, %s", exc)
                pass
            except pyodbc.Error as exc:
                logger.warning("Errors in checking geometries, %s", str(exc.args[1]))
                pass
        # pylint: disable=W0511
        # TODO: This produces wrong dtypes for an empty DataFrame.
        # We can't rely on type inference for an empty DataFrame;
        # we will need to provide dtype information explicitly.
        # The pandas.DataFrame constructor's dtype argument only allows
        # a single dtype for the whole DataFrame, but we should be able
        # to use the astype method instead.
        if not self.connection_context.pyodbc_connection:
            try:
                results = self.__run_query(select_statement, fetch_size)
            except dbapi.Error as exc:
                logger.error("Failed to retrieve data for the current dataframe, %s", exc)
                raise
            except pyodbc.Error as exc:
                logger.error("Failed to retrieve data for the current dataframe, %s", str(exc.args[1]))
                raise
            result_df = pd.DataFrame(results, columns=self.columns)
        else:
            result_df = pd.read_sql(select_statement, self.connection_context.connection)

        # Convert geometries to shapely objects in the pandas data frame,
        # so that they can be directly used in other python frameworks
        if not self.connection_context.pyodbc_connection:
            try:
                if geometries and convert_geo_to_shapely:
                    for col in geo_cols:
                        if result_df[col].isnull().values.any():
                            logger.warning("Column '%s' does contain Null values and can't be converted to a Shapely object. Will remain to be String.", col)
                        else:
                            result_df[col] = result_df[col].apply(wkt.loads)
            except dbapi.Error as exc:
                logger.warning("Errors in Converting geometries to shapely objects, %s", exc)
                pass
            except pyodbc.Error as exc:
                logger.warning("Errors in Converting geometries to shapely objects, %s", str(exc.args[1]))
                pass

        return result_df

    @property
    def geometries(self) -> list:
        """
        Returns the geometries of a data frame. The list is empty if there
        are none.

        Returns
        -------
        list
            List with geometry columns
        """
        res_list = []
        try:
            res_list = [col[0] for col in self.dtypes() if col[1] in ["GEOMETRY", "POINT"]]
        except Exception as err:
            logger.warning(err)
            pass
        return res_list

    @property
    def srids(self) -> dict:
        """
        Returns the srid for each geometry column in the dataframe.
        If none is found, the dictionary will be {}.

        For dataframes based on HANA catalog objects, the information is
        read from the catalog. For Dataframes, which do not have a catalog
        object (e.g. are based on SQL statements, or temporary tables ),
        the SRID is derived by selecting the first row in the table and
        read it directly from the EWKB. For columns with multiple SRSes
        (SRID NULL), this means, that you might get back a SRS, which differs
        from other entries in the same column.

        **Known Limitation**: For dataframes which don't have catalog objects
        and do not contain data, no SRID can be provided.

        Returns
        -------
        dict
            Dictionary with the SRID per column: `{<column_name>: <srid>}`
            Returns `{}` when none are found.
        """

        def get_from_data_row():
            """Helper to derive the SRID from a table row"""
            srid = {}

            # Get the relevant columns from the data source
            geo_cols_sql = ""
            geo_cols_list = []
            for col in self.dtypes():
                if col[1] in ["GEOMETRY", "POINT"]:
                    geo_cols_sql = ", ".join(
                        (geo_cols_sql, "{}.ST_SRID() ".format(quotename(col[0])))
                    )
                    geo_cols_list.append(col[0])
            geo_cols_sql = geo_cols_sql[2:]

            if geo_cols_sql == "":  # No Geometries Found
                return srid

            try:  # to get one record with values for the geo columns
                loc_result = self.__run_query(
                    "SELECT TOP 1 {} FROM({})".format(
                        geo_cols_sql, self.select_statement
                    )
                )

                # Turn the result into a dictionary
                if len(loc_result) == 1:
                    for index, col in enumerate(loc_result[0]):
                        try:
                            col_srid = int(col)
                            srid[geo_cols_list[index]] = col_srid
                        except Exception as err:
                            logger.warning(err)
                            pass
                else:
                    # Return a dictionary with geo cols and None Value
                    srid = dict.fromkeys(geo_cols_list, None)

            except Exception as ex:
                logger.error(str(ex))
                raise ex

            return srid

        # Read from DB for catalog objects
        if self.source_table:
            sql = """
                    SELECT COLUMN_NAME, SRS_ID
                      FROM ST_GEOMETRY_COLUMNS
                     WHERE SCHEMA_NAME = '{}'
                       AND TABLE_NAME = '{}'
                  """.format(self.source_table["SCHEMA_NAME"], self.source_table["TABLE_NAME"])
            result = self.__run_query(sql)

            # Fallback to reading the SRID from the table data (if there are any)
            if len(result) == 0:  # Will for example happen for temp. tables
                return get_from_data_row()
            else:
                return dict(result)
        else:  # Fallback to reading the SRID from the table data (if there are any)
            return get_from_data_row()

    def rename_columns(self, names):
        """
        Returns a DataFrame with renamed columns.

        Parameters
        ----------
        names : list or dict
            If a list, specifies new names for every column in this DataFrame.
            If a dict, each dict entry maps an existing name to a new name,
            and not all columns need to be renamed.

        Returns
        -------
        DataFrame
            The same data as the original DataFrame with new column names.

        See Also
        --------
        DataFrame.alias : For renaming the DataFrame itself.

        Examples
        --------

        >>> df.collect()
           A  B
        0  1  3
        1  2  4

        >>> df.rename_columns(['C', 'D']).collect()
           C  D
        0  1  3
        1  2  4

        >>> df.rename_columns({'B': 'D'}).collect()
           A  D
        0  1  3
        1  2  4
        """

        if isinstance(names, list):
            if len(names) != len(self.columns):
                if len(names) > len(self.columns):
                    problem = "Too many"
                else:
                    problem = "Not enough"
                raise ValueError(problem + ' columns in rename_columns list.')
            names = dict(zip(self.columns, names))
        elif isinstance(names, dict):
            bad_names = set(names).difference(self.columns)
            if bad_names:
                raise ValueError("Column(s) not in DataFrame: {}".format(
                    sorted(bad_names)))
        else:
            raise TypeError("names should be a list or dict, not {}".format(
                type(names)))
        retval = self.select(*[(quotename(orig), names[orig]) if orig in names
                               else orig
                               for orig in self.columns])
        self._propagate_safety(retval)
        return retval

    def cast(self, cols, new_type=None):
        """
        Returns a DataFrame with columns cast to a new type.

        The name of the column in the returned DataFrame is the same as the original column.
         .. warning::
           Type compatibility between the existing column type and the new type is not checked.
           An incompatibility results in an error.

        Parameters
        ----------
        cols : str, list of str or dict
            The column(s) to be cast to a different type.
        new_type : str
            The database datatype to cast the column(s) to.
            No checks are performed to see if the new type is valid.
            An invalid type can lead to SQL errors or even SQL injection vulnerabilities.

        Returns
        -------
        DataFrame
            The same data as this DataFrame, but with columns cast to the specified type.

        Examples
        --------
        Input:

        >>> df1 = cc.sql('SELECT "AGE", "PDAYS", "HOUSING" FROM DBM_TRAINING_TBL')
        >>> df1.dtypes()
        [('AGE', 'INT', 10, 10, 10, 0), ('PDAYS', 'INT', 10, 10, 10, 0), ('HOUSING', 'VARCHAR', 100, 100, 100, 0)]

        Casting a column to NVARCHAR(20):

        >>> df2 = df1.cast('AGE', 'NVARCHAR(20)')
        >>> df2.dtypes()
        [('AGE', 'NVARCHAR', 20, 20, 20, 0), ('PDAYS', 'INT', 10, 10, 10, 0), ('HOUSING', 'VARCHAR', 100, 100, 100, 0)]

        Casting a list of columns to NVARCHAR(50):

        >>> df3 = df1.cast(['AGE', 'PDAYS'], 'NVARCHAR(50)')
        >>> df3.dtypes()
        [('AGE', 'NVARCHAR', 50, 50, 50, 0), ('PDAYS', 'NVARCHAR', 50, 50, 50, 0), ('HOUSING', 'VARCHAR', 100, 100, 100, 0)]
        >>> df4 = df1.cast({'AGE': 'VARCHAR(50)', {'PDAYS': 'INT'}})
        >>> df4.dtypes()
        [('AGE', 'VARCHAR', 50, 50, 50, 0), ('PDAYS', 'INT', 50, 50, 50, 0), ('HOUSING', 'VARCHAR', 100, 100, 100, 0)]
        """
        if isinstance(cols, dict):
            projection = [quotename(col) if not col in cols\
                else 'CAST({0} AS {1}) AS {0}'.format(quotename(col), cols[col])\
                    for col in self.columns]
        else:
            cols = DataFrame._get_list_of_str(cols,
                                            'Parameter cols must be a string or a list of strings')
            self._validate_columns_in_dataframe(cols)

            # Need to check for valid types in newtype, skip for now
            projection = [quotename(col) if not col in cols\
                else 'CAST({0} AS {1}) AS {0}'.format(quotename(col), new_type)\
                    for col in self.columns]

        cast_select = 'SELECT {} FROM ({}) AS {}'.format(', '.join(projection),
                                                         self.select_statement,
                                                         self.quoted_name)
        return self._df(cast_select, propagate_safety=True)

    def to_head(self, col):
        """
        Returns a DataFrame with specified column as the first item in the projection.

        Parameters
        ----------
        col : str
            The column to move to the first position.

        Returns
        -------
        DataFrame
            The same data as this DataFrame but with the specified column in the first position.

        Examples
        --------
        Input:

        >>> df1 = cc.table("DBM_TRAINING")
        >>> import pprint
        >>> pprint.pprint(df1.columns)
        ['ID',
         'AGE',
         'JOB',
         'MARITAL',
         'EDUCATION',
         'DBM_DEFAULT',
         'HOUSING',
         'LOAN',
         'CONTACT',
         'DBM_MONTH',
         'DAY_OF_WEEK',
         'DURATION',
         'CAMPAIGN',
         'PDAYS',
         'PREVIOUS',
         'POUTCOME',
         'EMP_VAR_RATE',
         'CONS_PRICE_IDX',
         'CONS_CONF_IDX',
         'EURIBOR3M',
         'NREMPLOYED',
         'LABEL']

        Moving the column 'LABEL' to head:

        >>> df2 = df1.to_head('LABEL')
        >>> pprint.pprint(df2.columns)
        ['LABEL',
         'ID',
         'AGE',
         'JOB',
         'MARITAL',
         'EDUCATION',
         'DBM_DEFAULT',
         'HOUSING',
         'LOAN',
         'CONTACT',
         'DBM_MONTH',
         'DAY_OF_WEEK',
         'DURATION',
         'CAMPAIGN',
         'PDAYS',
         'PREVIOUS',
         'POUTCOME',
         'EMP_VAR_RATE',
         'CONS_PRICE_IDX',
         'CONS_CONF_IDX',
         'EURIBOR3M',
         'NREMPLOYED']
        """

        if not isinstance(col, _STRING_TYPES):
            raise TypeError('Parameter col must be a string')

        if col not in self.columns:
            raise ValueError('Column {} is not a column in this DataFrame'.format(quotename(col)))
        cols = self.columns
        cols.insert(0, cols.pop(cols.index(col)))
        return self[cols]

    def describe(self, cols=None):
        # The disable of line lengths is for the example in docstring.
        # The example is copy pasted after a run and may result in the output not quite lining up
        """
        Returns a DataFrame that contains various statistics for the requested column(s).

        Parameters
        ----------
        cols : str or list, optional
            The column(s) to be described. Defaults to all columns.

        Returns
        -------
        DataFrame
            A DataFrame that contains statistics for the specified column(s)
            in the current DataFrame.

            The statistics included are the count of rows ("count"),

            number of distinct values ("unique"),

            number of nulls ("nulls"),

            average ("mean"),

            standard deviation("std")

            median ("median"),

            minimum value ("min"),

            maximum value ("max"),

            25% percentile when treated as continuous variable ("25_percent_cont"),

            25% percentile when treated as discrete variable ("25_percent_disc"),

            50% percentile when treated as continuous variable ("50_percent_cont"),

            50% percentile when treated as discrete variable ("50_percent_disc"),

            75% percentile when treated as continuous variable ("75_percent_cont"),

            75% percentile when treated as discrete variable ("75_percent_disc").

            For columns that are strings, statistics such as average ("mean"),
            standard deviation ("std"), median ("median"), and the various percentiles
            are NULLs.

            If the list of columns contain both string and numeric data types,
            minimum and maximum values become NULLs.

        Examples
        --------
        Input:

        >>> df1 = cc.table("DBM_TRAINING")
        >>> import pprint
        >>> pprint.pprint(df2.columns)
        ['LABEL',
         'ID',
         'AGE',
         'JOB',
         'MARITAL',
         'EDUCATION',
         'DBM_DEFAULT',
         'HOUSING',
         'LOAN',
         'CONTACT',
         'DBM_MONTH',
         'DAY_OF_WEEK',
         'DURATION',
         'CAMPAIGN',
         'PDAYS',
         'PREVIOUS',
         'POUTCOME',
         'EMP_VAR_RATE',
         'CONS_PRICE_IDX',
         'CONS_CONF_IDX',
         'EURIBOR3M',
         'NREMPLOYED']

        Describe a few numeric columns and collect them to return a Pandas DataFrame:

        >>> df1.describe(['AGE', 'PDAYS']).collect()
          column  count  unique  nulls        mean         std  min  max  median
        0    AGE  16895      78      0   40.051376   10.716907   17   98      38
        1  PDAYS  16895      24      0  944.406688  226.331944    0  999     999
           25_percent_cont  25_percent_disc  50_percent_cont  50_percent_disc
        0             32.0               32             38.0               38
        1            999.0              999            999.0              999
           75_percent_cont  75_percent_disc
        0             47.0               47
        1            999.0              999

        Describe some non-numeric columns and collect them to return a Pandas DataFrame:

        >>> df1.describe(['JOB', 'MARITAL']).collect()
            column  count  unique  nulls  mean   std       min      max median
        0      JOB  16895      12      0  None  None    admin.  unknown   None
        1  MARITAL  16895       4      0  None  None  divorced  unknown   None
          25_percent_cont 25_percent_disc 50_percent_cont 50_percent_disc
        0            None            None            None            None
        1            None            None            None            None
          75_percent_cont 75_percent_disc
        0            None            None
        1            None            None

        Describe all columns in a DataFrame:

        >>> df1.describe().collect()
                    column  count  unique  nulls          mean           std
        0               ID  16895   16895      0  21282.286652  12209.759725
        1              AGE  16895      78      0     40.051376     10.716907
        2         DURATION  16895    1267      0    263.965670    264.331384
        3         CAMPAIGN  16895      35      0      2.344658      2.428449
        4            PDAYS  16895      24      0    944.406688    226.331944
        5         PREVIOUS  16895       7      0      0.209529      0.539450
        6     EMP_VAR_RATE  16895      10      0     -0.038798      1.621945
        7   CONS_PRICE_IDX  16895      26      0     93.538844      0.579189
        8    CONS_CONF_IDX  16895      26      0    -40.334123      4.865720
        9        EURIBOR3M  16895     283      0      3.499297      1.777986
        10      NREMPLOYED  16895      11      0   5160.371885     75.320580
        11             JOB  16895      12      0           NaN           NaN
        12         MARITAL  16895       4      0           NaN           NaN
        13       EDUCATION  16895       8      0           NaN           NaN
        14     DBM_DEFAULT  16895       2      0           NaN           NaN
        15         HOUSING  16895       3      0           NaN           NaN
        16            LOAN  16895       3      0           NaN           NaN
        17         CONTACT  16895       2      0           NaN           NaN
        18       DBM_MONTH  16895      10      0           NaN           NaN
        19     DAY_OF_WEEK  16895       5      0           NaN           NaN
        20        POUTCOME  16895       3      0           NaN           NaN
        21           LABEL  16895       2      0           NaN           NaN
                 min        max     median  25_percent_cont  25_percent_disc
        0      5.000  41187.000  21786.000        10583.500        10583.000
        1     17.000     98.000     38.000           32.000           32.000
        2      0.000   4918.000    184.000          107.000          107.000
        3      1.000     43.000      2.000            1.000            1.000
        4      0.000    999.000    999.000          999.000          999.000
        5      0.000      6.000      0.000            0.000            0.000
        6     -3.400      1.400      1.100           -1.800           -1.800
        7     92.201     94.767     93.444           93.075           93.075
        8    -50.800    -26.900    -41.800          -42.700          -42.700
        9      0.634      5.045      4.856            1.313            1.313
        10  4963.000   5228.000   5191.000         5099.000         5099.000
        11       NaN        NaN        NaN              NaN              NaN
        12       NaN        NaN        NaN              NaN              NaN
        13       NaN        NaN        NaN              NaN              NaN
        14       NaN        NaN        NaN              NaN              NaN
        15       NaN        NaN        NaN              NaN              NaN
        16       NaN        NaN        NaN              NaN              NaN
        17       NaN        NaN        NaN              NaN              NaN
        18       NaN        NaN        NaN              NaN              NaN
        19       NaN        NaN        NaN              NaN              NaN
        20       NaN        NaN        NaN              NaN              NaN
        21       NaN        NaN        NaN              NaN              NaN
            50_percent_cont  50_percent_disc  75_percent_cont  75_percent_disc
        0         21786.000        21786.000        32067.500        32068.000
        1            38.000           38.000           47.000           47.000
        2           184.000          184.000          324.000          324.000
        3             2.000            2.000            3.000            3.000
        4           999.000          999.000          999.000          999.000
        5             0.000            0.000            0.000            0.000
        6             1.100            1.100            1.400            1.400
        7            93.444           93.444           93.994           93.994
        8           -41.800          -41.800          -36.400          -36.400
        9             4.856            4.856            4.961            4.961
        10         5191.000         5191.000         5228.000         5228.000
        11              NaN              NaN              NaN              NaN
        12              NaN              NaN              NaN              NaN
        13              NaN              NaN              NaN              NaN
        14              NaN              NaN              NaN              NaN
        15              NaN              NaN              NaN              NaN
        16              NaN              NaN              NaN              NaN
        17              NaN              NaN              NaN              NaN
        18              NaN              NaN              NaN              NaN
        19              NaN              NaN              NaN              NaN
        20              NaN              NaN              NaN              NaN
        21              NaN              NaN              NaN              NaN
        """

        #pylint:disable=too-many-locals

        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns

        dtypes = self.dtypes(cols)
        # Not sure if this is the complete list but should cover most data types
        # Note that we don't cover BLOB/LOB types
        numerics = [col_info[0] for col_info in dtypes
                    if col_info[1] == "INT" or
                    col_info[1] == 'TINYINT' or
                    col_info[1] == 'BIGINT' or
                    col_info[1] == 'INTEGER' or
                    col_info[1] == 'DOUBLE' or
                    col_info[1] == 'DECIMAL' or
                    col_info[1] == 'FLOAT']
        non_numerics = [col_info[0] for col_info in dtypes
                        if col_info[1] == "NCHAR" or
                        col_info[1] == 'NVARCHAR' or
                        col_info[1] == 'CHAR' or
                        col_info[1] == 'VARCHAR' or
                        col_info[1] == 'STRING' or
                        col_info[1] == 'TIMESTAMP' or
                        col_info[1] == 'DATE']

        # The reason to separate numerics and non-numerics is the calculation
        # of min and max. These functions are of different types when calculating
        # for different types. So, a numeric type will return a numeric value
        # while a character column will return a character value for min and max
        # When both column types are present, the values for min and max would be null
        # for non numeric types
        sql_numerics = None
        sql_non_numerics = None
        min_max = 'MIN({0}) as "min", MAX({0}) as "max", '
        if numerics:
            sql_for_numerics = ('select {3} as "column", COUNT({0}) as "count", ' +
                                'COUNT(DISTINCT {0}) as "unique", ' +
                                'SUM(CASE WHEN {0} is NULL THEN 1 ELSE 0 END) as "nulls", ' +
                                'AVG(TO_DOUBLE({0})) as "mean", STDDEV({0}) as "std", ' +
                                min_max + 'MEDIAN({0}) as "median" ' +
                                'FROM ({1}) AS {2}')
            union = [sql_for_numerics.format(quotename(col), self.select_statement,
                                             self.quoted_name,
                                             "'{}'".format(col.replace("'", "''")))
                     for col in numerics]
            sql_simple_stats = " UNION ALL ".join(union)

            percentiles = ('SELECT {3} as "column", * FROM (SELECT ' +
                           'percentile_cont(0.25) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "25_percent_cont", '+
                           'percentile_disc(0.25) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "25_percent_disc", '+
                           'percentile_cont(0.50) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "50_percent_cont", '+
                           'percentile_disc(0.50) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "50_percent_disc", '+
                           'percentile_cont(0.75) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "75_percent_cont", '+
                           'percentile_disc(0.75) WITHIN GROUP (ORDER BY {0}) ' +
                           'AS "75_percent_disc" '+
                           'FROM ({1}) AS {2})')
            union = [percentiles.format(quotename(col), self.select_statement,
                                        self.quoted_name,
                                        "'{}'".format(col.replace("'", "''"))) for col in numerics]
            sql_percentiles = " UNION ALL ".join(union)

            sql_numerics = ('SELECT {0}.*, '.format('"SimpleStats"') +
                            ', '.join(['{0}."{1}_percent_cont", {0}."{1}_percent_disc"'.
                                       format('"Percentiles"', percentile)
                                       for percentile in [25, 50, 75]]) +
                            ' FROM ({0}) AS {1}, ({2}) AS {3}'.
                            format(sql_simple_stats, '"SimpleStats"',
                                   sql_percentiles, '"Percentiles"') +
                            ' WHERE {0}."column" = {1}."column"'.
                            format('"SimpleStats"', '"Percentiles"'))
            # This is to handle the case for non-numerics since min and max values
            # are now not compatible between numerics and non-numerics
            min_max = 'CAST(NULL AS DOUBLE) AS "min", CAST(NULL AS DOUBLE) AS "max", '

        if non_numerics:
            sql_for_non_numerics = ('select {3} as "column", COUNT({0}) as "count", ' +
                                    'COUNT(DISTINCT {0}) as "unique", ' +
                                    'SUM(CASE WHEN {0} IS NULL THEN 1 ELSE 0 END) as "nulls", ' +
                                    'CAST(NULL as DOUBLE) AS "mean", ' +
                                    'CAST(NULL as double) as "std", ' +
                                    'CAST(NULL as DOUBLE) AS "min", ' +
                                    'CAST(NULL as DOUBLE) AS "max", ' +
                                    'CAST(NULL as DOUBLE) AS "median", ' +
                                    'CAST(NULL AS DOUBLE) AS "25_percent_cont", ' +
                                    'CAST(NULL AS DOUBLE) AS "25_percent_disc", ' +
                                    'CAST(NULL AS DOUBLE) AS "50_percent_cont", ' +
                                    'CAST(NULL AS DOUBLE) AS "50_percent_disc", ' +
                                    'CAST(NULL AS DOUBLE) AS "75_percent_cont", ' +
                                    'CAST(NULL AS DOUBLE) AS "75_percent_disc" ' +
                                    'FROM ({1}) AS {2}')
            union = [sql_for_non_numerics.format(quotename(col), self.select_statement,
                                                 self.quoted_name,
                                                 "'{}'".format(col.replace("'", "''")))
                     for col in non_numerics]
            sql_non_numerics = " UNION ALL ".join(union)

        if sql_numerics is None and sql_non_numerics is None:
            raise ValueError('Parameter cols cannot be described')

        if sql_numerics is not None and sql_non_numerics is not None:
            sql_combined = ('SELECT * FROM ({0}) AS {1} UNION ALL SELECT * FROM ({2}) AS {3}'.
                            format(sql_numerics,
                                   '"Numerics"',
                                   sql_non_numerics,
                                   '"NonNumerics"'))
            return self._df(sql_combined, propagate_safety=True)
        if sql_numerics is not None:
            return self._df(sql_numerics, propagate_safety=True)

        return self._df(sql_non_numerics, propagate_safety=True)

    def bin(self, col, strategy='uniform_number', bins=None, bin_width=None,  #pylint: disable=too-many-arguments
            bin_column='BIN_NUMBER'):
        """
        Returns a DataFrame with the original columns as well as bin assignments.

        The name of the columns in the returned DataFrame is the same as the
        original column. Column "BIN_NUMBER" or the specified value in
        ``bin_column`` is added and corresponds to the bin assigned.

        Parameters
        ----------
        col : str
            The column on which binning is performed.
            The column must be numeric.
        strategy : {'uniform_number', 'uniform_size'}, optional
            Binning methods:

                - 'uniform_number': Equal widths based on the number of bins.
                - 'uniform_size': Equal widths based on the bin size.

            Default value is 'uniform_number'.

        bins : int, optional
            The number of equal-width bins.
            Only valid when ``strategy`` is 'uniform_number'.

            Defaults to 10.

        bin_width : int, optional
            The interval width of each bin.
            Only valid when ``strategy`` is 'uniform_size'.

        bin_column : str, optional
            The name of the output column that contains the bin number.

        Returns
        -------
        DataFrame
            A binned dataset with the same data as this DataFrame,
            as well as an additional column "BIN_NUMBER" or the value specified
            in ``bin_column``. This additional column contains the
            assigned bin for each row.

        Examples
        --------
        Input:

        >>> df.collect()
           C1   C2    C3       C4
        0   1  1.2   2.0      1.0
        1   2  1.4   4.0      3.0
        2   3  1.6   6.0      9.0
        3   4  1.8   8.0     27.0
        4   5  2.0  10.0     81.0
        5   6  2.2  12.0    243.0
        6   7  2.4  14.0    729.0
        7   8  2.6  16.0   2187.0
        8   9  2.8  18.0   6561.0
        9  10  3.0  20.0  19683.0

        Create five bins of equal widths on C1:

        >>> df.bin('C1', strategy='uniform_number', bins=5).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           2
        3   4  1.8   8.0     27.0           2
        4   5  2.0  10.0     81.0           3
        5   6  2.2  12.0    243.0           3
        6   7  2.4  14.0    729.0           4
        7   8  2.6  16.0   2187.0           4
        8   9  2.8  18.0   6561.0           5
        9  10  3.0  20.0  19683.0           5

        Create five bins of equal widths on C2:

        >>> df.bin('C3', strategy='uniform_number', bins=5).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           2
        3   4  1.8   8.0     27.0           2
        4   5  2.0  10.0     81.0           3
        5   6  2.2  12.0    243.0           3
        6   7  2.4  14.0    729.0           4
        7   8  2.6  16.0   2187.0           4
        8   9  2.8  18.0   6561.0           5
        9  10  3.0  20.0  19683.0           5

        Create five bins of equal widths on a column that varies significantly:

        >>> df.bin('C4', strategy='uniform_number', bins=5).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           1
        3   4  1.8   8.0     27.0           1
        4   5  2.0  10.0     81.0           1
        5   6  2.2  12.0    243.0           1
        6   7  2.4  14.0    729.0           1
        7   8  2.6  16.0   2187.0           1
        8   9  2.8  18.0   6561.0           2
        9  10  3.0  20.0  19683.0           5

        Create bins of equal width:

        >>> df.bin('C1', strategy='uniform_size', bin_width=3).collect()
           C1   C2    C3       C4  BIN_NUMBER
        0   1  1.2   2.0      1.0           1
        1   2  1.4   4.0      3.0           1
        2   3  1.6   6.0      9.0           2
        3   4  1.8   8.0     27.0           2
        4   5  2.0  10.0     81.0           2
        5   6  2.2  12.0    243.0           3
        6   7  2.4  14.0    729.0           3
        7   8  2.6  16.0   2187.0           3
        8   9  2.8  18.0   6561.0           4
        9  10  3.0  20.0  19683.0           4
        """

        if not isinstance(col, _STRING_TYPES):
            raise TypeError('Parameter col must be a string.')

        self._validate_columns_in_dataframe([col])

        if not self.is_numeric(col):
            raise ValueError('Parameter col must be a numeric column.')
        if not isinstance(strategy, _STRING_TYPES):
            raise TypeError('Parameter strategy must be a string')
        strategy = strategy.lower()
        if strategy not in ['uniform_number', 'uniform_size']:
            raise TypeError('Parameter strategy must be one of "uniform_number", "uniform_size".')

        if strategy == 'uniform_number':
            if bins is None:
                bins = 10
            if bin_width is not None:
                raise ValueError('Parameter bin_size invalid with strategy "uniform_number"')
        elif strategy == 'uniform_size':
            if (not isinstance(bin_width, _INTEGER_TYPES) and
                    not isinstance(bin_width, float)):
                raise TypeError('Parameter bin_width must be a numeric.')

        sql_template = 'SELECT *, BINNING(VALUE => {}, {} => {}) OVER() AS {} FROM ({}) AS {}'

        bin_select = sql_template.format(
            quotename(col),
            'BIN_COUNT' if strategy == 'uniform_number' else 'BIN_WIDTH',
            bins if strategy == 'uniform_number' else bin_width,
            quotename(bin_column), self.select_statement, self.quoted_name)
        return self._df(bin_select, propagate_safety=True)

    def agg(self, agg_list, group_by=None):
        """
        Returns a SAP HANA DataFrame with the group_by column along with the aggregates.
        This method supports all aggregation functions in the SAP HANA system, such as 'max', 'min',
        'count', 'avg', 'sum', 'median', 'stddev', 'var'.
        The name of the column in the returned DataFrame is the same as the
        original column.

        Aggregation functions can be referred to `SAP HANA aggregate functions\
        <https://help.sap.com/viewer/7c78579ce9b14a669c1f3295b0d8ca16/Cloud/en-US/6fff7f0ae9184d1db47a25791545a1b6.html>`_.

        Parameters
        ----------

        agg_list : A list of tuples

            A list of tuples. Each tuple is a triplet.
            The triplet consists of (aggregate_operator, expression, name) where:

                - aggregate_operator is one of ['max', 'min', 'count', 'avg',
                  'sum', 'median', 'stddev', 'var', ...].
                  The operator name is identical to SAP HANA sql naming and we support
                  all aggregation functions in the SAP HANA system.

                - expression is a str that is a column or column expression

                - name is the name of this aggregate in the project list.

        group_by : str or list of str, optional

            The group by column. Only a column is allowed although
            expressions are allowed in SQL. To group by an expression, create a
            DataFrame  by providing the entire SQL.
            So, if you have a table T with columns C1, C2, and C3 that are all
            integers, to calculate the max(C1) grouped by (C2+C3) a DataFrame
            would need to be created as below:

                cc.sql('SELECT "C2"+"C3", max("C1") FROM "T" GROUP BY "C2"+"C3"')

        Returns
        -------

        DataFrame
            A DataFrame containing the group_by column (if it exists), as well as
            the aggregate expressions that are aliased with the specified names.

        Examples
        --------

        Input:

        >>> df.collect()
            ID  SEPALLENGTHCM  SEPALWIDTHCM  PETALLENGTHCM  PETALWIDTHCM          SPECIES
        0    1            5.1           3.5            1.4           0.2      Iris-setosa
        1    2            4.9           3.0            1.4           0.2      Iris-setosa
        2    3            4.7           3.2            1.3           0.2      Iris-setosa
        3   51            7.0           3.2            4.7           1.4  Iris-versicolor
        4   52            6.4           3.2            4.5           1.5  Iris-versicolor
        5  101            6.3           3.3            6.0           2.5   Iris-virginica
        6  102            5.8           2.7            5.1           1.9   Iris-virginica
        7  103            7.1           3.0            5.9           2.1   Iris-virginica
        8  104            6.3           2.9            5.6           1.8   Iris-virginica

        Another way to do a count:

        >>> df.agg([('count', 'SPECIES', 'COUNT')]).collect()
            COUNT
        0      9

        Get counts by SPECIES:

        >>> df.agg([('count', 'SPECIES', 'COUNT')], group_by='SPECIES').collect()
                   SPECIES  COUNT
        0  Iris-versicolor      2
        1   Iris-virginica      4
        2      Iris-setosa      3

        Get max values of SEPALLENGTHCM by SPECIES:

        >>> df.agg([('max', 'SEPALLENGTHCM', 'MAX_SEPAL_LENGTH')], group_by='SPECIES').collect()
                   SPECIES  MAX_SEPAL_LENGTH
        0  Iris-versicolor               7.0
        1   Iris-virginica               7.1
        2      Iris-setosa               5.1

        Get max and min values of SEPALLENGTHCM by SPECIES:

        >>> df.agg([('max', 'SEPALLENGTHCM', 'MAX_SEPAL_LENGTH'),
            ('min', 'SEPALLENGTHCM', 'MIN_SEPAL_LENGTH')], group_by=['SPECIES']).collect()
                   SPECIES  MAX_SEPAL_LENGTH  MIN_SEPAL_LENGTH
        0  Iris-versicolor               7.0               6.4
        1   Iris-virginica               7.1               5.8
        2      Iris-setosa               5.1               4.7

        Get aggregate grouping by multiple columns:

        >>> df.agg([('count', 'SEPALLENGTHCM', 'COUNT_SEPAL_LENGTH')],
                    group_by=['SPECIES', 'PETALLENGTHCM']).collect()
                   SPECIES  PETALLENGTHCM  COUNT_SEPAL_LENGTH
        0   Iris-virginica            6.0                   1
        1      Iris-setosa            1.3                   1
        2   Iris-virginica            5.9                   1
        3   Iris-virginica            5.6                   1
        4      Iris-setosa            1.4                   2
        5  Iris-versicolor            4.7                   1
        6  Iris-versicolor            4.5                   1
        7   Iris-virginica            5.1                   1
        """

        if group_by is not None:
            msg = 'Parameter group_by must be a string or a list of strings.'
            group_by = DataFrame._get_list_of_str(group_by, msg)
            self._validate_columns_in_dataframe(group_by)
            group_by = [quotename(gb) for gb in group_by]

        if not isinstance(agg_list, list):
            raise TypeError('Parameter agg_list must be a list.')
        if not agg_list:
            raise ValueError('Parameter agg_list must contain at least one tuple.')
        aggregates = []
        for item in agg_list:
            if not isinstance(item, tuple):
                raise TypeError('Parameter agg_list must be a tuple with 3 elements.')
            if len(item) != 3:
                raise TypeError('Parameter agg_list must be a list of tuples with 3 elements.')
            (agg, expr, name) = item
            agg = agg.lower()
            aggregates.append('{}({}) AS {}'.format(agg, quotename(expr), quotename(name)))

        sql = ('SELECT {} {} FROM ({}) AS {}{}'
               .format('' if group_by is None else (','.join(group_by) + ','),
                       ', '.join(aggregates),
                       self.select_statement,
                       self.quoted_name,
                       '' if group_by is None
                       else ' GROUP BY {}'.format(','.join(group_by))))
        return self._df(sql)

    def is_numeric(self, cols=None):
        """
        Returns True if the column(s) in the DataFrame are numeric.

        Parameters
        ----------
        cols : str or list, optional
            The column(s) to be tested for being numeric.

            Defaults to all columns.

        Returns
        -------
        bool
            True if all the columns are numeric.

        Examples
        --------
        Input:

        >>> df.head(5).collect()
        ID  SEPALLENGTHCM  SEPALWIDTHCM  PETALLENGTHCM  PETALWIDTHCM      SPECIES
        0   1            5.1           3.5            1.4           0.2  Iris-setosa
        1   2            4.9           3.0            1.4           0.2  Iris-setosa
        2   3            4.7           3.2            1.3           0.2  Iris-setosa
        3   4            4.6           3.1            1.5           0.2  Iris-setosa
        4   5            5.0           3.6            1.4           0.2  Iris-setosa

        >>> pprint.pprint(df.dtypes())
        [('ID', 'INT', 10, 10, 10, 0),
         ('SEPALLENGTHCM', 'DOUBLE', 15, 15, 15, 0),
         ('SEPALWIDTHCM', 'DOUBLE', 15, 15, 15, 0),
         ('PETALLENGTHCM', 'DOUBLE', 15, 15, 15, 0),
         ('PETALWIDTHCM', 'DOUBLE', 15, 15, 15, 0),
         ('SPECIES', 'NVARCHAR', 15, 15, 15, 0)]

        Test a single column:

        >>> df.is_numeric('ID')
        True
        >>> df.is_numeric('SEPALLENGTHCM')
        True
        >>> df.is_numeric(['SPECIES'])
        False

        Test a list of columns:

        >>> df.is_numeric(['SEPALLENGTHCM', 'PETALLENGTHCM', 'PETALWIDTHCM'])
        True
        >>> df.is_numeric(['SEPALLENGTHCM', 'PETALLENGTHCM', 'SPECIES'])
        False
        """

        if cols is not None:
            msg = 'Parameter cols must be a string or a list of strings.'
            cols = DataFrame._get_list_of_str(cols, msg)
            self._validate_columns_in_dataframe(cols)
        else:
            cols = self.columns

        return all(col_info[1] in ['INT', 'TINYINT', 'BIGINT', 'SMALLINT',
                                   'INTEGER', 'DOUBLE', 'DECIMAL', 'FLOAT']
                   for col_info in self.dtypes(cols))

    def corr(self, first_col, second_col):
        """
        Returns a DataFrame that gives the correlation coefficient between two
        numeric columns.

        All rows with NULL values for first_col or second_col are removed
        prior to calculating the correlation coefficient.

        The correlation coefficient is:
            1/(n-1) * sum((col1_value - avg(col1)) * (col2 - avg(col2))) /
            (stddev(col1) * stddev(col2))

        Parameters
        ----------
        first_col : str
            The first column for calculating the correlation coefficient.
        second_col : str
            The second column for calculating the correlation coefficient.

        Returns
        -------
        DataFrame
            A DataFrame with one value that contains the correlation coefficient.
            The name of the column is CORR_COEFF.

        Examples
        --------
        Input:

        >>> df.columns
        ['C1', 'C2', 'C3', 'C4']

        >>> df.collect()
           C1   C2      C3       C4
        0   1  1.2     2.0      1.0
        1   2  1.4     4.0      3.0
        2   3  1.6     8.0      9.0
        3   4  1.8    16.0     27.0
        4   5  2.0    32.0     81.0
        5   6  2.2    64.0    243.0
        6   7  2.4   128.0    729.0
        7   8  2.6   256.0   2187.0
        8   9  2.8   512.0   6561.0
        9  10  3.0  1024.0  19683.0

        Correlation with columns that are well correlated:

        >>> df.corr('C1', 'C2').collect()
              CORR_COEFF
        0         1.0

        >>> df.corr('C1', 'C3').collect()
            CORR_COEFF
        0         1.0

        Correlation with a column whose value is three times its previous value:

        >>> df.corr('C1', 'C4').collect()
            CORR_COEFF
        0    0.696325
        """

        if not isinstance(first_col, _STRING_TYPES):
            raise TypeError('Parameter first_col must be a string.')
        if not isinstance(second_col, _STRING_TYPES):
            raise TypeError('Parameter second_col must be a string.')
        if not self.is_numeric([first_col, second_col]):
            raise ValueError('Correlation columns {0} and {1} must be numeric.'.
                             format(first_col, second_col))
        dt_sql = ('SELECT * FROM ({0}) AS {1} WHERE ' +
                  '{1}.{2} is not NULL and {1}.{3} is not NULL')
        derived_table = self._df(dt_sql.format(self.select_statement,
                                               self.quoted_name, quotename(first_col), quotename(second_col)))

        corr_select = ('select A.A/B.B AS CORR_COEFF from (select (sum(({0} - ' +
                       '(select avg({0}) from ({2}) )) * ({1} - ' +
                       '(select avg({1}) from ({2}) )))) as A from ({2})) A, ' +
                       '(select stddev({0})*stddev({1})*(count(*)-1) as B ' +
                       'from ({2})) B ').format(quotename(first_col), quotename(second_col),
                                                derived_table.select_statement)

        dfc = self._df(corr_select, propagate_safety=True)
        return dfc

    def min(self):
        """
        Get the minimum value of the columns. It simplies the use of agg function.

        Returns
        -------

            scalar or Series
        """
        if len(self.columns) > 1:
            agg_list = []
            for col in self.columns:
                agg_list.append(('min', col, col))
            return self.agg(agg_list).collect(geometries=False).iloc[0]
        else:
            return self.agg([('min', self.columns[0], self.columns[0])]).collect(geometries=False).iat[0, 0]

    def max(self):
        """
        Get the maximum value of the columns. It simplies the use of agg function.

        Returns
        -------

            scalar or Series
        """
        if len(self.columns) > 1:
            agg_list = []
            for col in self.columns:
                agg_list.append(('max', col, col))
            return self.agg(agg_list).collect(geometries=False).iloc[0]
        else:
            return self.agg([('max', self.columns[0], self.columns[0])]).collect(geometries=False).iat[0, 0]

    def sum(self):
        """
        Get the summation of the columns. It simplies the use of agg function.

        Returns
        -------

            scalar or Series
        """
        if len(self.columns) > 1:
            agg_list = []
            for col in self.columns:
                agg_list.append(('sum', col, col))
            return self.agg(agg_list).collect(geometries=False).iloc[0]
        else:
            return self.agg([('sum', self.columns[0], self.columns[0])]).collect(geometries=False).iat[0, 0]

    def median(self):
        """
        Get the median value of the columns. It simplies the use of agg function.

        Returns
        -------

            scalar or Series
        """
        if len(self.columns) > 1:
            agg_list = []
            for col in self.columns:
                agg_list.append(('median', col, col))
            return self.agg(agg_list).collect(geometries=False).iloc[0]
        else:
            return self.agg([('median', self.columns[0], self.columns[0])]).collect(geometries=False).iat[0, 0]

    def mean(self):
        """
        Get the mean value of the columns. It simplies the use of agg function.

        Returns
        -------

            scalar or Series
        """
        if len(self.columns) > 1:
            agg_list = []
            for col in self.columns:
                agg_list.append(('avg', col, col))
            return self.agg(agg_list).collect(geometries=False).iloc[0]
        else:
            return self.agg([('avg', self.columns[0], self.columns[0])]).collect(geometries=False).iat[0, 0]

    def value_counts(self, subset=None):
        """
        Get the value counts of the columns. It simplies the use of agg function.

        Parameters
        ----------
        subset : list, optional
            Columns to use when counting unique combinations.

        Returns
        -------

            DataFrame
        """
        if subset is None:
            subset = self.columns
        count_df = []
        id_df = []
        for col in subset:
            id_df.append(self.select(col).rename_columns({col: "VALUES"}).cast("VALUES", 'NVARCHAR(255)'))
            count_df.append(self.agg([("count", col, "NUM_{}".format(col))], group_by=col).cast(col, 'NVARCHAR(255)').set_index(col))
        idf = id_df[0].union(id_df[1:]).distinct().set_index("VALUES")
        return idf.join(count_df, how="left")

    def pivot_table(self, values, index, columns, aggfunc='avg'):
        """
        Returns a DataFrame that gives the pivoted table.

        ``aggfunc`` is identical to SAP HANA aggregate functions:

        https://help.sap.com/viewer/7c78579ce9b14a669c1f3295b0d8ca16/Cloud/en-US/6fff7f0ae9184d1db47a25791545a1b6.html

        Parameters
        ----------
        values : str or list of str
            The targeted values for pivoting.
        index : str or list of str
            The index of the DataFrame.
        columns : str or list of str
            The pivoting columns.
        aggfunc : {'avg', 'max', 'min',... }, optional
            ``aggfunc`` is identical to SAP HANA aggregate functions.
            Defaults to 'avg'.

        Returns
        -------
        DataFrame
            A pivoted DataFrame.

        Examples
        --------
        df is a SAP HANA DataFrame.

        >>> df.pivot_table(values='C2', index='C1', columns='C3', aggfunc='max')

        """
        columns_tmp = None
        index_tmp = None
        if isinstance(columns, list):
            columns_tmp = " || '_' || ".join(quotename(col) for col in columns)
        else:
            columns_tmp = quotename(columns)
        if isinstance(index, str) is False:
            index_tmp = ", ".join(quotename(col) for col in index)
        else:
            index_tmp = quotename(index)
        col_set = self.__run_query('SELECT ' + ' distinct ' + columns_tmp + ' FROM (' + self.select_statement + ')')
        col_set = set(map(lambda x: str(x[0]), col_set))
        sql_script = 'SELECT ' + index_tmp + ', '
        for col in col_set:
            if not isinstance(values, list):
                sql_script = sql_script + aggfunc + '(CASE WHEN ' + columns_tmp + \
                '=' + "'" + col + "'" + ' THEN ' + quotename(values) + ' END) AS ' + quotename(col) + ','
            else:
                for val in values:
                    sql_script = sql_script + aggfunc + '(CASE WHEN ' + columns_tmp + \
                    '=' + "'" + col + "'" + ' THEN ' + quotename(val) + ' END) AS ' + quotename(col + '|' + val) + ','
        sql_script = sql_script[:-1]
        sql_script = sql_script + ' FROM (' + self.select_statement + ') GROUP BY ' + index_tmp
        return self._df(sql_script)

    # SQLTRACE
    def generate_table_type(self):
        """
        Generates a SAP HANA table type based on the dtypes function of the DataFrame. This is a convenience method for SQL tracing.

        Parameters
        ----------
        None

        Returns
        -------
        str
            The table type in string form.

        """

        dtypes = self.dtypes()
        type_string = "" + "("
        first = True
        for rows in dtypes:
            if first:
                first = False
            else:
                type_string = type_string + ","
            if rows[1] in ('INT', 'INTEGER', 'DOUBLE', 'SMALLINT', 'TINYINT', 'BIGINT', 'CLOB', 'BLOB'):
                type_part = "\"" + rows[0] +"\""  + " " + rows[1]
                type_string = type_string + type_part
            if rows[1] == 'VARCHAR' or rows[1] == 'NVARCHAR':
                type_part = "\"" + rows[0] +"\" " + rows[1] + "(" + str(rows[2]) + ")"
                type_string = type_string + type_part
            if rows[1] == 'DECIMAL':
                type_part = "\"" + rows[0] +"\" " + rows[1] + "(" + str(rows[4]) + "," + str(rows[5]) + ")"
                type_string = type_string + type_part
        type_string = type_string + ")"
        table_types = "table {}".format(type_string)
        return table_types

    def set_source_table(self, table, schema=None):
        """
        Specify the source table for the current dataframe.

        Parameters
        ----------
        table : str
            The table name.
        schema : str, optional, keyword-only
            The schema name. If this value is not provided or set to None, then the value defaults to the
            ConnectionContext's current schema.
        """
        if schema is None:
            try:
                schema = self.connection_context.get_current_schema()
            except:
                schema = "DM_PAL_FOR_UNITTEST"
        self.source_table = {"SCHEMA_NAME": schema, "TABLE_NAME": table}

    def to_pickle(self, path, compression='infer', protocol=4):
        """
        Pickle object to file.

        Parameters
        ----------
        path : str
            File path where the pickled object will be stored.
        compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
            A string representing the compression to use in the output file. By default, infers from the file extension in specified path.

            Defaults to 'infer'.
        protocol : int
            Int which indicates which protocol should be used by the pickler, default HIGHEST_PROTOCOL.
        """
        pandas_df = self.collect()
        pandas_df.to_pickle(path, compression, protocol)

def read_pickle(connection_context, path, table_name, compression='infer', schema=None, force=False, replace=True, object_type_as_bin=False, table_structure=None, **kwargs):
    """
    Load pickled DataFrame object from file.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.
    path : str
        File path where the pickled object will be loaded.
    table_name : str
        The table name in the SAP HANA database.
    compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}, optional
        For on-the-fly decompression of on-disk data. If 'infer', then use gzip, bz2, xz or zip if path ends in '.gz', '.bz2', '.xz', or '.zip' respectively, and no decompression otherwise. Set to None for no decompression.

        Defaults to 'infer'.
    schema : str, optional, keyword-only
        The schema name. If this value is not provided or set to None, then the value defaults to the
        ConnectionContext's current schema.

        Defaults to the current schema.
    force : bool, optional
        If force is True, then the SAP HANA table with table_name is dropped.

        Defaults to False.
    replace : bool, optional
        If replace is True, then the SAP HANA table performs the missing value handling.

        Defaults to True.
    object_type_as_bin : bool, optional
        If True, the object type will be considered CLOB in SAP HANA.

        Defaults to False.
    table_structure : dict, optional
        Manually define column types based on SAP HANA DB table structure.

        Defaults to None.
    """
    pandas_df = pd.read_pickle(path, compression)
    return create_dataframe_from_pandas(connection_context, pandas_df, table_name, schema, force, replace, object_type_as_bin, table_structure, **kwargs)

#Internal
def export_into(connection_context, tables, condition="", directory=None, export_format="BINARY", replace=False, scramble=False, threads=1):
    """
    Export SAP HANA tables into the disk.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.
    tables : str or list of str
        Table(s) ready for export.
    condition : str, optional
        Exports a subset of table data. The WHERE clause follows, and is associated with, a single table in the EXPORT statement.

        Defaults to "".
    directory : str, optional
        Export location.

        Defaults to '/tmp/HANAML_<uuid>.tgz'.
    export_format : {"BINARY, "CSV"}, optional

        Defaults to "BINARY".
    replace : bool, optional
        Defines the behavior if the export data already exists in the specified directory. If REPLACE is not specified, then an error is returned if previously exported data exists in the specified export directory.

        Defaults to False.
    scramble : bool, optional
        Obfuscates CSV format exported data.

        Defaults to False.
    threads : int
        The number of threads used for export.
    """
    if isinstance(tables, str):
        tables = [tables]
    if directory is None:
        directory = "/tmp/HANAML_{}.tgz".format(uuid.uuid1())
    export_options = []
    if replace is True:
        export_options.append("REPLACE")
    if scramble is True:
        export_options.append("SCRAMBLE")
    if threads > 1:
        export_options.append("THREADS {}".format(threads))
    options = ""
    if len(export_options) > 1:
        options = "WITH {}".format(" ".join(export_options))
    query = "EXPORT {0} {1} AS {2} INTO '{3}' {4}".format(", ".join(tables), condition, export_format, directory, options)
    with connection_context.connection.cursor() as cur:
        execute_logged(cur,
                       query,
                       connection_context.sql_tracer,
                       connection_context)
    if not connection_context.pyodbc_connection:
        if not connection_context.connection.getautocommit():
            connection_context.connection.commit()
    return directory

#Internal
def import_from(connection_context, directory, replace=False, threads=1):
    """
    Import data into the SAP HANA system.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.

    directory : str
        Specifies the location where the import source is found. Specify <archive_file_name> if the import data is in an archive file. The archive file must have the file extension .tar.gz or .tgz.

    replace : bool, optional
        Defines the behavior if the import data already exists in the database. When specified, if a table defined in the import data currently exists in the database, then it is dropped and recreated before the data is imported. If the REPLACE option is not specified, then an error is thrown if an existing database table is defined in the import data.

        Defaults to False.
    threads : int, optional
        The number of threads used for import.

        Defaults to 1.
    """
    import_options = []
    if replace is True:
        import_options.append("REPLACE")
    if threads > 1:
        import_options.append("THREADS {}".format(threads))
    options = ""
    if len(import_options) > 1:
        options = "WITH {}".format(" ".join(import_options))
    query = "IMPORT ALL FROM {0} {1}".format(directory, options)
    with connection_context.connection.cursor() as cur:
        execute_logged(cur,
                       query,
                       connection_context.sql_tracer,
                       connection_context)
    if not connection_context.pyodbc_connection:
        if not connection_context.connection.getautocommit():
            connection_context.connection.commit()

def data_manipulation(connection_context, table_name, unload=True, schema=None, persistent_memory=None):
    """
    Load/unload the data from memory within the SAP HANA database environment. Note that this method has no impact on the client.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.

    table_name : str
        The table name in the SAP HANA database

    unload : bool, optional
        - True : Unload the data from memory.
        - False : Load the data from memory.

        Defaults to True.
    schema : str, optional
        The schema name.

        Defaults to None.
    persistent_memory : {'retain', 'delete'}, optional
        Only works when persistent memory is enabled.

        Defaults to None.

    Returns
    -------
    None

    Examples
    --------
    >>> data_manipulation(conn, 'TEST_TBL')
    """
    load_sql = "LOAD {} all".format(quotename(table_name))
    if schema is not None:
        load_sql = "LOAD {}.{} all".format(quotename(schema), quotename(table_name))
    if unload is True:
        load_sql = "UNLOAD {}".format(quotename(table_name))
        if schema is not None:
            load_sql = "UNLOAD {}.{}".format(quotename(schema), quotename(table_name))
        if persistent_memory is not None:
            load_sql = load_sql + " {} PERSISTENT MEMORY".format(persistent_memory.upper())
    with connection_context.connection.cursor() as cur:
        execute_logged(cur,
                       load_sql,
                       connection_context.sql_tracer,
                       connection_context)
    if not connection_context.pyodbc_connection:
        if not connection_context.connection.getautocommit():
            connection_context.connection.commit()


def _get_dtype(dtype, allow_bigint, object_type_as_bin):
    """
    Return a string representing the SAP HANA data type of a given Pandas data type. Used by create_pandas_dataframe in the
    case a primary key column or not null column is being written.

    Parameters
    ----------
    dtype : str
        A connection to the SAP HANA system.

    allow_bigint : bool
        allow_bigint decides whether int64 is mapped into INT or BIGINT in SAP HANA.

    object_type_as_bin : bool
        If True, the object type will be considered CLOB in SAP HANA.

    Returns
    -------
    str

    """
    if ('int64' in dtype.lower()) and allow_bigint:
        return 'BIGINT'
    elif 'int' in dtype.lower():
        return 'INT'
    elif 'float' in dtype.lower() or 'double' in dtype.lower():
        return 'DOUBLE'
    elif 'datetime' in dtype.lower():
        return 'TIMESTAMP'
    elif 'object' in dtype.lower() and object_type_as_bin:
        return 'CLOB'
    else:
        return 'NVARCHAR(255)'


def _prepare_lat_lon_for_import(pandas_df: pd.DataFrame, lon_lat: tuple):
    """ Helper to prepare the latitude/longitude columns for import"""
    column_name = "{lon}_{lat}_GEO".format(lon=lon_lat[0], lat=lon_lat[1])

    # Add the point as WKT to the dataframe
    pandas_df[column_name] = "POINT(" + pandas_df[lon_lat[0]].astype(str).str.cat(
        pandas_df[lon_lat[1]].astype(str) + ")", sep=" "
    )

    # Replace cells with invalid geometry
    pandas_df.loc[
        pandas_df[column_name] == "POINT(nan nan)", [column_name]
    ] = "POINT EMPTY"

    return column_name, pandas_df


def _prepare_geo_cols_for_import(pandas_df: pd.DataFrame, col: str):
    """ Helper to convert the different geometry formats to WKT for downstream processing"""
    if len(pandas_df) > 0:
        sample_value = pandas_df.iloc[0][col]

        if isinstance(sample_value, memoryview):
            pass  # Nothing to do will be handled by HANA

        elif str(type(sample_value))[:25] == "<class 'shapely.geometry.":
            pandas_df[col] = pandas_df[col].apply(
                lambda val: np.nan if val is None or isinstance(val, float) else val.wkt
            )

    column_name = "{}_GEO".format(col)
    pandas_df = pandas_df.rename(columns={col: column_name})

    return column_name, pandas_df


def create_dataframe_from_pandas(
        connection_context,
        pandas_df,
        table_name,
        schema=None,
        force=False,
        replace=False,
        object_type_as_bin=False,
        table_structure=None,
        drop_exist_tab=True,
        allow_bigint=False,
        geo_cols: list = None,  # Spatial variable
        srid: int = 4326,  # Spatial variable
        primary_key: str = None,  # Graph variable
        not_nulls: list = None,  # Graph variable
        chunk_size=50000,
        disable_progressbar=False,
):
    """
    Uploads data from a Pandas DataFrame to a SAP HANA database and returns an SAP HANA DataFrame.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.
    pandas_df : pandas.DataFrame
        A Pandas DataFrame for uploading to the SAP HANA database. This can
        also be a GeoPandas datframe, which will automatically be converted
        to a Pandas DataFrame
    table_name : str
        The table name in the SAP HANA database.
    schema : str, optional, keyword-only
        The schema name. If this value is not provided or set to None, then the value defaults to the
        ConnectionContext's current schema.

        Defaults to the current schema.
    force : bool, optional
        If force is True, then the SAP HANA table with table_name is truncated or dropped.

        Defaults to False.
    replace : bool, optional
        If replace is True, then the SAP HANA table performs the missing value handling.

        Defaults to False.
    object_type_as_bin : bool, optional
        If True, the object type will be considered CLOB in SAP HANA.

        Defaults to False.
    table_structure : dict
        Manually define column types based on SAP HANA DB table structure.
    drop_exist_tab : bool, optional
        If force is True, drop the existing table when drop_exist_tab is True and truncate the existing table when it is False.

        Defaults to True.
    allow_bigint : bool, optional
        allow_bigint decides whether int64 is mapped into INT or BIGINT in HANA.

        Defaults to False.
    geo_cols : list, optional but required for spatial functions
        Specifies the columns of the dataframe, which are treated as geometries.
        List elements can be either strings or tuples.

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
    srid : int, optional but required for spatial functions
        Spatial reference system id. If the SRS is not created yet, the
        system tries to create it automatically.

        Defaults to 4326.
    primary_key : str, optional but required for Graph functions
        Name of the column in a node table which contains the unique identification of the node and corresponds with the
        edge table.

        Defaults to None.
    not_nulls : list, optional but required for Graph functions
        Contains all column names which should get a not null constraint. This is primarily for creating node and edge
        tables for a graph workspace

        Defaults to None.
    chunk_size : int, optional
        Specify the chunk size for upload.

        Defaults to 50000.
    disable_progressbar : bool, optional
        Disable the progressbar.

        Defaults to False.

    Returns
    -------
    DataFrame
        A SAP HANA DataFrame that contains the data in the pandas_df.

    Examples
    --------
    >>> create_dataframe_from_pandas(connection_context,p_df,'test',force=False,replace=True,drop_exist_tab=False)
    <hana_ml.dataframe.DataFrame at 0x7efbcb26fbe0>

    >>> create_dataframe_from_pandas(
            connection_context,
            p_df,
            "geo_table",
            force=False,
            replace=True,
            drop_exist_tab=False,
            geo_cols=["geometry", ("long", "lat")],
        )
    <hana_ml.dataframe.DataFrame at 0x5eabcb27fbe0>
    """
    # Initialized list makes life easier
    if not geo_cols:
        geo_cols = []

    # Turn geopandas df in a pandas df
    # checking by isinstance requires importing the geopandas so check
    # for a string to change into standard pandas
    if str(type(pandas_df)) == "<class 'geopandas.geodataframe.GeoDataFrame'>":
        # Automatically add the geopandas geometry to the list of geometries,
        # if it's not already in
        if pandas_df.geometry.name not in geo_cols:
            geo_cols.append(pandas_df.geometry.name)

        pandas_df = pd.DataFrame(pandas_df)

    # Check if the parameters are correct and convert geo columns

    prepared_geo_cols = []  # List of ge columns in the df, after they are prepared for import. Base for further processing.
    if not isinstance(geo_cols, list):
        raise ValueError("geo_cols needs to be a list of columns")

    for col in geo_cols:
        if isinstance(col, tuple):
            if len(col) < 2:
                raise ValueError("Column names for both, latitude and longitude are required")

            if col[0] not in pandas_df.columns:
                raise ValueError("Column '{}' not found in dataframe".format(col[0]))

            if col[1] not in pandas_df.columns:
                raise ValueError("Column '{}' not found in dataframe".format(col[1]))

            geo_col_name, pandas_df = _prepare_lat_lon_for_import(pandas_df, col)
            prepared_geo_cols.append(geo_col_name)

        elif isinstance(col, str):
            if col not in pandas_df.columns:
                raise ValueError("Column '{}' not found in dataframe".format(col))

            geo_col_name, pandas_df = _prepare_geo_cols_for_import(pandas_df, col)
            prepared_geo_cols.append(geo_col_name)

        else:
            raise ValueError("Unsupported datatype of '{}'".format(col))

    # Check if the SRS is already created and try to create it if not
    if prepared_geo_cols:
        from .spatial import create_predefined_srs  # pylint: disable=import-outside-toplevel
        create_predefined_srs(connection_context=connection_context, srid=srid)

    if schema is None:
        table_reference = quotename(table_name)
    else:
        table_reference = '.'.join(map(quotename, (schema, table_name)))
    cursor = connection_context.connection.cursor()
    tab_exist = connection_context.has_table(table=table_name, schema=schema)

    if force is True:
        sql_script = 'DROP TABLE {};'.format(table_reference)
        if (tab_exist == True) and (drop_exist_tab == False):
            sql_script = 'TRUNCATE TABLE {};'.format(table_reference)
            logger.info("Table already exists. Begin to truncate table.")
        # execute drop table with try catch
        try:
            execute_logged(cursor,
                           sql_script,
                           connection_context.sql_tracer,
                           connection_context)
        except dbapi.Error as db_er:# pylint: disable=unused-variable
            pass
        except pyodbc.Error as db_er:# pylint: disable=unused-variable
            pass
    # create table script
    if (force == False) and (tab_exist == True):
        logger.warning("Table already exists. Data will be appended to the existing table. To set `force=True` can empty the table.")
    replace_subset = []

    if (tab_exist == False) or ((tab_exist == True) and (drop_exist_tab == True)):
        sql_script = 'CREATE TABLE {} ('.format(table_reference)
        if '#' in table_reference[:2]:
            sql_script = 'CREATE LOCAL TEMPORARY TABLE {} ('.format(table_reference)
        dtypes_list = list(map(str, pandas_df.dtypes.values))

        for col, dtype in zip(pandas_df.columns, dtypes_list):
            if table_structure is not None and col in table_structure:
                sql_script = sql_script + '"{}" {}, '.format(col, table_structure[col])
            else:
                # Adjust the column dtype with a primary key and not_null in the case it is required
                if primary_key and col == primary_key:
                    sql_key = ' primary key,'
                    sql_null = ''
                elif not_nulls and col in not_nulls:
                    sql_key = ''
                    sql_null = ' not null,'
                else:
                    # primary key takes precendence over not_null and therefore takes the , if neither is exists for this column
                    sql_key = ','
                    sql_null = ''

                # Limit the sql_key to a NVARCHAR 255
                if sql_key not in ['', ',']:
                    dtype_col = _get_dtype(dtype=dtype, allow_bigint=allow_bigint,
                                           object_type_as_bin=object_type_as_bin)
                    sql_script = sql_script + '"{}" {}{}{} '.format(col, dtype_col, sql_key, sql_null)
                elif sql_null != '':
                    dtype_col = _get_dtype(dtype=dtype, allow_bigint=allow_bigint,
                                           object_type_as_bin=object_type_as_bin)
                    sql_script = sql_script + '"{}" {}{}{} '.format(col, dtype_col, sql_key, sql_null)
                elif col in prepared_geo_cols:
                    sql_script = sql_script + '"{}" ST_GEOMETRY({}), '.format(col, srid)
                elif ('int64' in dtype.lower()) and allow_bigint:
                    sql_script = sql_script + '"{}" {}, '.format(col, 'BIGINT')
                    replace_subset.append(col)
                elif 'int' in dtype.lower():
                    sql_script = sql_script + '"{}" {}, '.format(col, 'INT')
                    replace_subset.append(col)
                elif 'float' in dtype.lower() or 'double' in dtype.lower():
                    sql_script = sql_script + '"{}" {}, '.format(col, 'DOUBLE')
                    replace_subset.append(col)
                elif 'datetime' in dtype.lower():
                    sql_script = sql_script + '"{}" {}, '.format(col, 'TIMESTAMP')
                elif 'object' in dtype.lower() and object_type_as_bin:
                    sql_script = sql_script + '"{}" {}, '.format(col, 'CLOB')
                else:
                    sql_script = sql_script + '"{}" {}, '.format(col, 'VARCHAR(5000)')

        sql_script = sql_script[:-2]
        sql_script = sql_script + ');'
        try:
            execute_logged(cursor,
                           sql_script,
                           connection_context.sql_tracer,
                           connection_context)
        except dbapi.Error as db_er:
            logger.error(str(db_er))
            cursor.close()
            raise
        except pyodbc.Error as db_er:
            logger.error(str(db_er.args[1]))
            cursor.close()
            raise
    if pandas_df.isnull().values.any() and replace:
        logger.info("Replace nan with 0 in numeric columns.")

    shape0 = len(pandas_df)

    if shape0 > 0:
        # Prepare the columns for insert statement
        parms = ""
        for col in pandas_df.columns:
            if col in prepared_geo_cols:
                parms = ','.join((parms, 'ST_GEOMFROMWKT(?, {})'.format(srid)))
            else:
                parms = ','.join((parms, '?'))

        parms = parms[1:]

        sql = 'INSERT INTO {} VALUES ({})'.format(table_reference, parms)
        num_regular_chunks = math.floor(float(shape0)/float(chunk_size))
        cum = 0
        has_dtype_time = "time" in str(pandas_df.dtypes).lower()
        has_dtype_numpy = "numpy" in " ".join([str(dtype.type) for dtype in pandas_df.dtypes]).lower()

        if has_dtype_time and has_dtype_numpy:
            for chunk in tqdm(range(1, num_regular_chunks + 2), disable=disable_progressbar):
                begin = cum
                cum = cum + chunk_size

                if chunk <= num_regular_chunks:
                    rows = tuple(map(tuple, [[None if element is None or pd.isnull(element) else (str(element) if 'time' in str(type(element)) else (element.item() if (type(element).__module__ == np.__name__) else element)) for element in lines] for lines in pandas_df.iloc[begin:cum].values])) #pylint: disable=line-too-long
                    cursor.executemany(sql, rows)
                else:
                    rows = tuple(map(tuple, [[None if element is None or pd.isnull(element) else (str(element) if 'time' in str(type(element)) else (element.item() if (type(element).__module__ == np.__name__) else element)) for element in lines] for lines in pandas_df.iloc[begin:].values])) #pylint: disable=line-too-long
                    if rows:
                        cursor.executemany(sql, rows)
        elif has_dtype_numpy:
            for chunk in tqdm(range(1, num_regular_chunks + 2), disable=disable_progressbar):
                begin = cum
                cum = cum + chunk_size

                if chunk <= num_regular_chunks:
                    rows = tuple(map(tuple, [[None if element is None or pd.isnull(element) else (element.item() if (type(element).__module__ == np.__name__) else element) for element in lines] for lines in pandas_df.iloc[begin:cum].values])) #pylint: disable=line-too-long
                    cursor.executemany(sql, rows)
                else:
                    rows = tuple(map(tuple, [[None if element is None or pd.isnull(element) else (element.item() if (type(element).__module__ == np.__name__) else element) for element in lines] for lines in pandas_df.iloc[begin:].values])) #pylint: disable=line-too-long
                    if rows:
                        cursor.executemany(sql, rows)
        else:
            for chunk in tqdm(range(1, num_regular_chunks + 2), disable=disable_progressbar):
                begin = cum
                cum = cum + chunk_size

                if chunk <= num_regular_chunks:
                    rows = [tuple(x) for x in pandas_df.iloc[begin:cum].where(pd.notnull(pandas_df), None).values]
                    cursor.executemany(sql, rows)
                else:
                    rows = [tuple(x) for x in pandas_df.iloc[begin:].where(pd.notnull(pandas_df), None).values]
                    if rows:
                        cursor.executemany(sql, rows)

    cursor.close()
    if not connection_context.pyodbc_connection:
        if not connection_context.connection.getautocommit():
            connection_context.connection.commit()

    res_df = DataFrame(connection_context, 'SELECT * FROM {}'.format(table_reference))
    res_df.set_source_table(table_name, schema)

    if replace:
        return res_df.fillna(0, subset=replace_subset)

    return res_df


def melt(frame, id_vars=None, value_vars=None, var_name=None, value_name=None):
    """
    Unpivot a DataFrame from wide format to long format, optionally leaving identifier variables set.

    Parameters
    ----------
    frame : DataFrame
        A SAP HANA DataFrame.

    id_vars : str, tuple or list, optional
        Column(s) to use as identifier variables.

        Defaults to None.

    value_vars : tuple or list, optional
        Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars.

    var_name : scalar, optional
        Name to use for the 'variable' column. If None it uses frame.columns.name or 'variable'.

    value_name : scalar, default 'value', optional
        Name to use for the 'value' column.

    Returns
    -------
    DataFrame
        Unpivoted DataFrame.

    Examples
    --------
    >>> data.collect()
        A B C
     0  a 1 2
     1  b 3 4
     2  c 5 6

    >>> melt(data, id_vars=['A'], value_vars=['B', 'C']).collect()
        A variable value
     0  a        B     1
     1  a        C     2
     2  b        B     5
     3  b        C     6
     4  c        B     3
     5  c        C     4
    """
    if value_vars is None:
        if id_vars is None:
            value_vars = frame.columns
        if isinstance(id_vars, (list, tuple)):
            value_vars = [element for element in frame.columns if element not in id_vars]
        if isinstance(id_vars, str):
            value_vars = [element for element in frame.columns if element != id_vars]
    if isinstance(id_vars, (list, tuple)):
        id_vars = ", ".join(id_vars)
    if not isinstance(value_vars, (list, tuple)):
        value_vars = [value_vars]
    if var_name is None:
        var_name = "variable"
    if value_name is None:
        value_name = "value"
    var_list = []
    for var in value_vars:
        if id_vars is not None:
            var_list.append("SELECT {0}, '{1}' AS \"{2}\", {1} AS \"{3}\" FROM ({4})"
                            .format(id_vars,
                                    var,
                                    var_name,
                                    value_name,
                                    frame.select_statement))
        else:
            var_list.append("SELECT '{0}' AS \"{1}\", {0} AS \"{2}\" FROM ({3})"
                            .format(var,
                                    var_name,
                                    value_name,
                                    frame.select_statement))
    exec_sql = ' UNION '.join(var_list)
    if len(value_vars) < 1:
        exec_sql = frame.select_statement
    return DataFrame(frame.connection_context, exec_sql)


def create_dataframe_from_shapefile(
        connection_context: ConnectionContext,
        shp_file: str,
        srid: int,
        table_name: str,
        schema: str = None,
) -> DataFrame:
    """
    Given a shapefile change the file into a DataFrame so that it is
    backed in SAP HANA. Expects that the shapefile name is a zip and/or
    will have both shp and dbf parts to create the single table. Once the
    table is created temporarily and locally, import the table as a
    shapefile into the target table through direct insertion. Then return
    the SAP HANA Dataframe that can be visualized.

    Parameters
    ----------
    connection_context : ConnectionContext
        A connection to the SAP HANA system.

    shp_file : str
        Path to a zipfile, shapefile or dbf-file. Filename suffix will
        be dropped and attempt to load both dbf and shp by that base name.
        Allowed suffixes are: .zip, .shp, .shx, and .dbf

    srid : int
        The spatial reference id that applies to the list of columns in geo_cols.
        If the SRS is not created yet, the system tries to create it automatically.

    table_name : str
        The table name in the SAP HANA database.

    schema : str, optional, keyword-only
        The schema name. If this value is not provided or set to None,
        then the value defaults to the ConnectionContext's current schema.

        Defaults to the current schema.

    Returns
    -------
    DataFrame
        A SAP HANA DataFrame with geometric columns that contains the data
        from the shp_file.

    Examples
    --------
    >>> cc = connection_context
    >>> shapefile_path = os.path.join(os.getcwd(), 'myshape.shp')
    >>> hana_df = create_dataframe_from_shapefile(
    ...   connection_context=cc,
    ...   shp_file=shapefile_path,
    ...   srid=4326,
    ...   table_name="myshape_tbl")
    """

    def log_and_raise_value_error(message):
        logger.error(message)
        raise ValueError(message)

    file_base, file_extension = os.path.splitext(shp_file)

    shp_file_path = ""
    dbf_file_path = ""

    if file_extension == ".zip":
        # Check if the file exists
        if not os.path.isfile(shp_file):
            log_and_raise_value_error(
                "Archive '{filename}' was not found.".format(filename=shp_file)
            )

        # Extract the shapefile and the dbf file from the zip archive
        with ZipFile(shp_file, "r") as zip_file:
            # Search the .shp file in the zip-file and extract it
            try:
                shp_file_path = [
                    zip_file.extract(shp_file_name, os.path.dirname(shp_file))
                    for shp_file_name in zip_file.namelist()
                    if shp_file_name[-4:] == ".shp"
                ][0]
            except IndexError:
                log_and_raise_value_error("No .shp file found in archive.")

            # Search the .dbf file in the zip-file and extract it
            try:
                dbf_file_path = [
                    zip_file.extract(shp_file_name, os.path.dirname(shp_file))
                    for shp_file_name in zip_file.namelist()
                    if shp_file_name[-4:] == ".dbf"
                ][0]
            except IndexError:
                log_and_raise_value_error("No .dbf file found in archive.")

    elif file_extension in [".dbf", ".shp", ".shx"]:
        # These are the two mandatory files we need, derived from the
        # filename without extension
        shp_file_path = file_base + ".shp"
        dbf_file_path = file_base + ".dbf"

        # Check if the files exist
        if not os.path.isfile(shp_file_path):
            log_and_raise_value_error(
                "'{filename}' was not found.".format(filename=shp_file_path)
            )

        if not os.path.isfile(dbf_file_path):
            log_and_raise_value_error(
                "'{filename}' was not found.".format(filename=dbf_file_path)
            )

    else:
        log_and_raise_value_error(
            "'{extension}' is an invalid file type.".format(extension=file_extension)
        )

    logger.info("Importing %s", shp_file_path)
    logger.info("Importing %s", dbf_file_path)

    # Check if the SRS is already created and try to create it
    from .spatial import create_predefined_srs  # pylint: disable=import-outside-toplevel
    create_predefined_srs(connection_context=connection_context, srid=srid)

    # Load the files to an temporary HANA DB table
    shp_file_handle = open(shp_file_path, "rb")
    dbf_file_handle = open(dbf_file_path, "rb")

    if not schema:
        schema = connection_context.get_current_schema()

    table_reference = '"{schema}"."{table}"'.format(schema=schema, table=table_name)

    # Drop the local and target tables in the case they exist
    try:
        connection_context.connection.cursor().execute("DROP TABLE #IMPORT_TABLE")
    except dbapi.ProgrammingError:
        # Silently pass the does not exist (invalid table name)
        pass

    try:
        connection_context.connection.cursor().execute(
            "DROP TABLE {}".format(table_reference)
        )
    except dbapi.ProgrammingError:
        # Silently pass the does not exist (invalid table name)
        pass

    try:
        with connection_context.connection.cursor() as cursor:
            cursor.execute(
                """
            CREATE LOCAL TEMPORARY TABLE #IMPORT_TABLE (
                FILENAME VARCHAR(64),
                PATH VARCHAR(255),
                CONTENT BLOB)
            """
            )

            cursor.execute(
                "INSERT INTO #IMPORT_TABLE VALUES (?,'',?)",
                (table_name + ".shp", shp_file_handle.read()),
            )

            cursor.execute(
                "INSERT INTO #IMPORT_TABLE VALUES (?,'',?)",
                (table_name + ".dbf", dbf_file_handle.read()),
            )

            cursor.execute(
                """
                IMPORT {table_reference} AS SHAPEFILE FROM #IMPORT_TABLE WITH SRID {srid}
                """.format(
                    table_reference=table_reference, srid=srid
                )
            )

    except dbapi.ProgrammingError as exception:
        logger.error(exception)
        raise exception

    finally:
        shp_file_handle.close()
        dbf_file_handle.close()

    return DataFrame(
        connection_context,
        "SELECT * FROM {}".format(table_reference),
    )
