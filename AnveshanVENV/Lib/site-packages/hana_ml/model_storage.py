#pylint: disable=too-many-lines
# pylint: disable=unused-import
"""
This module provides the features of **model storage**.

All these features are accessible to the end user via ModelStorage class:

    * :class:`ModelStorage`
    * :class:`ModelStorageError`

"""
import logging
import time
import datetime
import importlib
import json
import os
import signal
import socket
from multiprocessing import Process

import pandas as pd
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
import schedule
import hana_ml
from hana_ml.ml_exceptions import Error
from hana_ml.dataframe import DataFrame, quotename, ConnectionContext
from hana_ml.ml_base import execute_logged, try_drop

logging.basicConfig()
logger = logging.getLogger(__name__) #pylint: disable=invalid-name

# pylint: disable=too-few-public-methods
# pylint: disable=protected-access
# pylint: disable=too-many-statements
# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=eval-used
# pylint: disable=unused-argument
# pylint: disable=raise-missing-from
# pylint: disable=consider-using-f-string

class ModelStorageError(Error):
    """Exception class used in Model Storage"""

class ModelStorage(object): #pylint: disable=useless-object-inheritance
    """
    The ModelStorage class allows users to **save**, **list**, **load** and
    **delete** models.

    Models are saved into SAP HANA tables in a schema specified by the user.
    A model is identified with:

    - A name (string of 255 characters maximum),
      It must not contain any characters such as coma, semi-colon, tabulation, end-of-line,
      simple-quote, double-quote (',', ';', '"', ''', '\\n', '\\t').
    - A version (positive integer starting from 1).

    A model can be saved in three ways:

    1) It can be saved for the first time.
       No model with the same name and version is supposed to exist.
    2) It can be saved as a replacement.
       If a model with the same name and version already exists, it could be overwritten.\n
    3) It can be saved with a higher version.
       The model will be saved with an incremented version number.

    Internally, a model is stored as two parts:

    1) The metadata.
       It contains the model identification (name, version, algorithm class) and also its
       python model object attributes required for reinstantiation.
       It is saved in a table named **HANAML_MODEL_STORAGE** by default.
    2) The back-end model.
       It consists in the model returned by SAP HANA APL or SAP HANA PAL.
       For SAP HANA APL, it is always saved into the table **HANAMl_APL_MODELS_DEFAULT**,
       while for SAP HANA PAL, a model can be saved into different tables depending on the nature of the
       specified algorithm.

    Parameters
    ----------
    connection_context : ConnectionContext
        The connection object to a SAP HANA database.
        It must be the same as the one used by the model.

    schema : str, optional
        The schema name where the model storage tables are created.

        Defaults to the current schema used by the user.

    meta : str, optional
        The name of meta table stored in SAP HANA.

        Defaults to 'HANAML_MODEL_STORAGE'.


    Examples
    --------

    Creating and training a model with functions MLPClassifier and AutoClassifier:

    Assume the training data is data and the connection to SAP HANA is conn.

    >>> model_pal_name = 'MLPClassifier 1'
    >>> model_pal = MLPClassifier(conn, hidden_layer_size=[10, ], activation='TANH', output_activation='TANH', learning_rate=0.01, momentum=0.001)
    >>> model_pal.fit(data, label='IS_SETOSA', key='ID')

    >>> model_apl_name = 'AutoClassifier 1'
    >>> model_apl = AutoClassifier(conn_context=conn)
    >>> model_apl.fit(data, label='IS_SETOSA', key='ID')

    Creating an instance of ModelStorage:

    >>> MODEL_SCHEMA = 'MODEL_STORAGE' # HANA schema in which models are to be saved
    >>> model_storage = ModelStorage(connection_context=conn, schema=MODEL_SCHEMA)

    Saving these two trained models for the first time:

    >>> model_pal.name = model_pal_name
    >>> model_storage.save_model(model=model_pal)
    >>> model_apl.name = model_apl_name
    >>> model_storage.save_model(model=model_apl)

    Listing saved models:

    >>> print(model_storage.list_models())
                   NAME  VERSION LIBRARY                         ...
    0  AutoClassifier 1        1     APL  hana_ml.algorithms.apl ...
    1  MLPClassifier 1         1     PAL  hana_ml.algorithms.pal ...

    Reloading saved models:

    >>> model1 = model_storage.load_model(name=model_pal_name, version=1)
    >>> model2 = model_storage.load_model(name=model_apl_name, version=1)

    Using loaded model model2 for new prediction:

    >>> out = model2.predict(data=data_test)
    >>> print(out.head(3).collect())
       ID PREDICTED  PROBABILITY IS_SETOSA
    0   1      True     0.999492      None    ...
    1   2      True     0.999478      None
    2   3      True     0.999460      None

    Other examples of functions:

    Saving a model by overwriting the original model:

    >>> model_storage.save_model(model=model_apl, if_exists='replace')
    >>> print(list_models = model_storage.list_models(name=model.name))
                   NAME  VERSION LIBRARY                            ...
    0  AutoClassifier 1        1     APL  hana_ml.algorithms.apl    ...

    Saving a model by upgrading the version:

    >>> model_storage.save_model(model=model_apl, if_exists='upgrade')
    >>> print(list_models = model_storage.list_models(name=model.name))
                   NAME  VERSION LIBRARY                            ...
    0  AutoClassifier 1        1     APL  hana_ml.algorithms.apl    ...
    1  AutoClassifier 1        2     APL  hana_ml.algorithms.apl    ...

    Deleting a model with specified version:

    >>> model_storage.delete_model(name=model.name, version=model.version)

    Deleteing models with same model name and different versions:

    >>> model_storage.delete_models(name=model.name)

    Clean up all models and meta data at once:

    >>> model_storage.clean_up()

    """

    _VERSION = 1
    # Metadata table
    _METADATA_TABLE_NAME = 'HANAML_MODEL_STORAGE'
    # to-do : add schedule time and dataset table for re-fitting in model storage
    _METADATA_TABLE_DEF = ('('
                           'NAME VARCHAR(255) NOT NULL, '
                           'VERSION INT NOT NULL, '
                           'LIBRARY VARCHAR(128) NOT NULL, '
                           'CLASS VARCHAR(255) NOT NULL, '
                           'JSON VARCHAR(5000) NOT NULL, '
                           'TIMESTAMP TIMESTAMP NOT NULL, '
                           'STORAGE_TYPE VARCHAR(255) NOT NULL, '
                           'MODEL_STORAGE_VER INT NOT NULL, '
                           'SCHEDULE VARCHAR(5000), '
                           'PRIMARY KEY(NAME, VERSION) )'
                          )
    _METADATA_NB_COLS = 9

    def __init__(self, connection_context, schema=None, meta=None):
        self.connection_context = connection_context
        if schema is None:
            schema = connection_context.sql("SELECT CURRENT_SCHEMA FROM DUMMY")\
            .collect()['CURRENT_SCHEMA'][0]
        self.schema = schema
        if meta is not None:
            self._METADATA_TABLE_NAME = meta
        self.schedule_config_template = {
            "schedule":
            {
                "status" : 'inactive',
                "schedule_time" : 'every 1 hours',
                "pid" : None,
                "client" : None,
                "connection" : {
                    "userkey" : 'your_userkey',
                    "encrypt" : 'false',
                    "sslValidateCertificate" : 'true'
                },
                "hana_ml_obj" : 'hana_ml.algorithms.pal.unified_classification.UnifiedClassification',
                "init_params" : {"func" : 'HybridGradientBoostingTree',
                                 "param_search_strategy" : 'grid',
                                 "resampling_method" : 'cv',
                                 "evaluation_metric" : 'error_rate',
                                 "ref_metric" : ['auc'],
                                 "fold_num" : 5,
                                 "random_state" : 1,
                                 "param_values" : {'learning_rate': [0.1, 0.4, 0.7, 1],
                                                   'n_estimators': [4, 6, 8, 10],
                                                   'split_threshold': [0.1, 0.4, 0.7, 1]}},
                "fit_params" : {"key" : 'ID',
                                "label" : 'CLASS',
                                "partition_method" : 'stratified',
                                "partition_random_state" : 1,
                                "stratified_column" : 'CLASS'},
                "training_dataset_select_statement" : 'SELECT * FROM YOUR_TABLE'
            }
        }
        self.logfile = 'schedule.log'
        self.data_lake_container = 'SYSRDL#CG'

    # ===== Methods callable by the end user

    def list_models(self, name=None, version=None):
        """
        Lists existing models.

        Parameters
        ----------
        name : str, optional
            The model name pattern to be matched.

            Defaults to None.
        version : int, optional
            The model version.

            Defaults to None.

        Returns
        -------
        pandas.DataFrame
            The model metadata matching the provided name and version.
        """
        if name:
            self._check_valid_name(name)
        if not self._check_metadata_exists():
            raise ModelStorageError('No model was saved (no metadata)')
        sql = "select * from {schema_name}.{table_name}".format(
            schema_name=quotename(self.schema),
            table_name=quotename(self._METADATA_TABLE_NAME))
        where = ''
        param_vals = []
        if name:
            where = "NAME like ?"
            param_vals.append(name)
        if version:
            if where:
                where = where + ' and '
            where = where + 'version = ?'
            param_vals.append(int(version))
        if where:
            sql = sql + ' where ' + where
        with self.connection_context.connection.cursor() as cursor:
            if where:
                logger.info("Executing SQL: %s %s", sql, str(param_vals))
                cursor.execute(sql, param_vals)
            else:
                logger.info("Executing SQL: %s", sql)
                cursor.execute(sql)
            res = cursor.fetchall()

            col_names = [i[0] for i in cursor.description]
            res = pd.DataFrame(res, columns=col_names)
        return res

    def model_already_exists(self, name, version):
        """
        Checks if a model with specified name and version already exists.

        Parameters
        ----------
        name : str
            The model name.
        version : int
            The model version.

        Returns
        -------
        bool
            If True, there is already a model with the same name and version.
            If False, there is no model with the same name.
        """
        self._check_valid_name(name)
        if not self._check_metadata_exists():
            return False
        with self.connection_context.connection.cursor() as cursor:
            try:
                sql = "select count(*) from {schema_name}.{table_name} " \
                      "where NAME = ? " \
                      "and version = ?".format(
                          schema_name=quotename(self.schema),
                          table_name=quotename(self._METADATA_TABLE_NAME))
                params = [name, int(version)]
                logger.info('Execute SQL: %s %s', sql, str(params))
                cursor.execute(sql, params)
                res = cursor.fetchall()
                if res[0][0] > 0:
                    return True
            except dbapi.Error as err:
                logger.warning(err)
                pass
            return False

    def save_model(self, model, if_exists='upgrade', storage_type='default', force=False):
        """
        Saves a model.

        Parameters
        ----------
        model : a model instance.
            The model name must have been set before saving the model.
            The information of name and version will serve as an unique id of a model.
        if_exists : str, optional
            Specifies the behavior how a model is saved if a model with same name/version already exists:
                - 'fail': Raises an Error.
                - 'replace': Overwrites the previous model.
                - 'upgrade': Saves the model with an incremented version.

            Defaults to 'upgrade'.
        storage_type : {'default', 'HDL'}, optional
            Specifies the storage type of the model:
                - 'default' : HANA default storage.
                - 'HDL' : HANA data lake.
        force : bool, optional
            Drop the existing table if True.
        """
        # Checks the artifact model table exists for the current model
        if not model.is_fitted() and model._is_APL():
            # Problem with PAL; like GLM, some models do not have model_ attribute
            raise ModelStorageError(
                "The model cannot be saved. Please fit the model or load an existing one.")
        # Checks the parameter if_exists is correct
        if if_exists not in {'fail', 'replace', 'upgrade'}:
            raise ValueError("Unexpected value of 'if_exists' parameter: ", if_exists)
        # Checks if the model name is set
        name, version = self._get_model_id(model)
        if not name or not version and model._is_APL():
            raise ModelStorageError('The name of the model must be set.')
        self._check_valid_name(name)
        model_id = {'name': name, 'version': version}
        # Checks a model with the same name already exists
        model_exists = self.model_already_exists(**model_id)
        if model_exists:
            if if_exists == 'fail':
                raise ModelStorageError('A model with the same name/version already exists')
            if if_exists == 'upgrade':
                version = self._get_new_version_no(name=name)
                setattr(model, 'version', version)
                model_id = {'name': name, 'version': version}

        # Starts transaction to save data
        # Sets autocommit to False to ensure transaction isolation
        conn = self.connection_context.connection # SAP HANA connection
        old_autocommit = conn.getautocommit()
        conn.setautocommit(False)
        logger.info("Executing SQL: -- Disable autocommit")
        try:
            # Disables autocommit for tables creation
            with self.connection_context.connection.cursor() as cursor:
                execute_logged(cursor, 'SET TRANSACTION AUTOCOMMIT DDL OFF',
                               None, self.connection_context)
            # Creates metadata table if it does not exist
            self._create_metadata_table()
            # Deletes old version for replacement
            if model_exists and if_exists == 'replace':
                # Deletes before resaving as a new model
                self._delete_model(name, version, self.data_lake_container)
            # Saves the back-end model and returns the json for metadata
            # pylint: disable=protected-access
            js_str = model._encode_and_save(schema=self.schema,
                                            storage_type=storage_type,
                                            force=force,
                                            data_lake_container=self.data_lake_container)
            # Saves metadata with json
            self._save_metadata(model=model, js_str=js_str, storage_type=storage_type)
            # commits changes in database
            logger.info('Executing SQL: commit')
            conn.commit()
        except dbapi.Error as db_er:
            logger.error("An issue occurred in database during model saving: %s",
                         db_er, exc_info=True)
            logger.info('Executing SQL: rollback')
            conn.rollback()
            raise ModelStorageError('Unable to save the model.')
        except Exception as ex:
            logger.error('An issue occurred during model saving: %s', ex, exc_info=True)
            logger.info('Executing SQL: rollback')
            conn.rollback()
            raise ModelStorageError('Unable to save the model')
        finally:
            logger.info('Model %s is correctly saved', model.name)
            conn.setautocommit(old_autocommit)

    def delete_model(self, name, version):
        """
        Deletes a model with a given name and version.

        Parameters
        ----------
        name : str
            The model name.

        version : int
            The model version.
        """
        if not name or not version:
            raise ValueError("The model name and version must be specified.")
        self._check_valid_name(name)
        if self.model_already_exists(name, version):
            # Sets autocommit to False to ensure transaction isolation
            conn = self.connection_context.connection  # SAP HANA connection
            old_autocommit = conn.getautocommit()
            conn.setautocommit(False)
            try:
                # Gets the json string fromp the metadata
                self._delete_model(name=name,
                                   version=version,
                                   data_lake_container=self.data_lake_container)
                logger.info('Executing SQL: commit')
                conn.commit()
            except dbapi.Error as db_er:
                logger.error("An issue occurred in database during model removal:: %s",
                             db_er, exc_info=True)
                logger.info('Executing SQL: rollback')
                conn.rollback()
                raise ModelStorageError('Unable to delete the model.')
            except Exception as ex:
                logger.error('An issue occurred during model removal: %s', ex, exc_info=True)
                logger.info('Executing SQL: rollback')
                conn.rollback()
                raise ModelStorageError('Unable to delete the model.')
            finally:
                logger.info('Model %s is correctly deleted.', name)
                conn.setautocommit(old_autocommit)
        else:
            raise ModelStorageError('There is no model/version with this name:', name)

    def delete_models(self, name, start_time=None, end_time=None):
        """
        Deletes the model in a batch model with specified time range.

        Parameters
        ----------
        name : str
            The model name.
        start_time : str, optional
            The start timestamp for deleting.

            Defaults to None.
        end_time : str, optional
            The end timestamp for deleting.

            Defaults to None.
        """
        if not name:
            raise ValueError("The model name must be specified.")
        self._check_valid_name(name)
        meta = self.list_models(name=name)
        if start_time is not None and end_time is not None:
            meta = meta[(meta['TIMESTAMP'] >= start_time) & (meta['TIMESTAMP'] <= end_time)]
        elif start_time is not None and end_time is None:
            meta = meta[meta['TIMESTAMP'] >= start_time]
        elif start_time is None and end_time is not None:
            meta = meta[meta['TIMESTAMP'] <= end_time]
        else:
            pass
        for row in meta.itertuples():
            self.delete_model(name, row.VERSION)

    def clean_up(self):
        """
        Be cautious! This function will delete all the models and the meta table.
        """
        if self._check_metadata_exists():
            try:
                for model in set(self.list_models()['NAME'].to_list()):
                    self.delete_models(model)
            except ModelStorageError as ms_err:
                logger.error(ms_err)
                pass
            try_drop(self.connection_context, self._METADATA_TABLE_NAME)

    def load_model(self, name, version=None, **kwargs):
        """
        Loads an existing model from the SAP HANA database.

        Parameters
        ----------
        name : str
            The model name.
        version : int, optional
            The model version.
            By default, the last version will be loaded.

        Returns
        -------
        PAL/APL object
            The loaded model ready for use.
        """
        self._check_valid_name(name)
        if not version:
            version = self._get_last_version_no(name=name)
        if not self.model_already_exists(name=name, version=version):
            raise ModelStorageError('The model "{}" version {} does not exist'.format(
                name, version))
        metadata = self._get_model_metadata(name=name, version=version)
        # pylint: disable=protected-access
        model_class = self._load_class(metadata['CLASS'])
        js_str = metadata['JSON']
        model = model_class._load_model(
            connection_context=self.connection_context,
            name=name,
            version=version,
            js_str=js_str,
            **kwargs)
        algorithms = ['ARIMA', 'AutoARIMA', 'VectorARIMA', 'OnlineARIMA']
        if type(model).__name__ in algorithms:
            model.set_conn(self.connection_context)
        return model

    # ===== Private methods

    @staticmethod
    def _load_class(class_full_name):
        """
        Imports the required module for <class_full_name> and returns the class.

        Parameters
        ----------
        class_full_name: str
            The fully qualified class name.

        Returns
        -------
        The class
        """
        components = class_full_name.split('.')
        if "src" in components:
            components.remove("src")
        mod_name = '.'.join(components[:-1])
        mod = importlib.import_module(mod_name)
        cur_class = getattr(mod, components[-1])
        return cur_class

    def _create_metadata_table(self):
        """"
        Creates the metadata table if it does not exists.
        """
        if not self._check_metadata_exists():
            with self.connection_context.connection.cursor() as cursor:
                sql = 'CREATE COLUMN TABLE {schema_name}.{table_name} {cols_def}'.format(
                    schema_name=quotename(self.schema),
                    table_name=quotename(self._METADATA_TABLE_NAME),
                    cols_def=self._METADATA_TABLE_DEF)
                execute_logged(cursor, sql, self.connection_context.sql_tracer, self.connection_context)

    def _save_metadata(self, model, js_str, schedule_config_str=None, storage_type='default'):
        """
        Saves the model metadata.

        Parameters
        ----------
        model : A SAP HANA ML model
            A model instance.
        js_str : str
            JSON string to be saved in the metadata table.
        schedule_config_str : str
            Scheduler config to be saved in the meatedata table.
        """
        if schedule_config_str is None:
            schedule_config_str = self.schedule_config_template
        with self.connection_context.connection.cursor() as cursor:
            # Inserts data into the metadata table
            now = time.time()  # timestamp
            now_str = datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')
            lib = 'PAL'  # lib
            fclass_name = model.__module__ + '.' + type(model).__name__ # class name
            if model.__module__.startswith('hana_ml.algorithms.apl'):
                lib = 'APL'
            sql = 'INSERT INTO {}.{} VALUES({})'.format(
                quotename(self.schema),
                quotename(self._METADATA_TABLE_NAME),
                ', '.join(['?'] * self._METADATA_NB_COLS)
                )
            logger.info("Prepare SQL: %s", sql)
            data = [(
                model.name,
                int(model.version),
                lib,
                fclass_name,
                js_str,
                now_str,
                storage_type,
                self._VERSION,
                json.dumps(schedule_config_str)
                )]
            logger.info("Executing SQL: INSERT INTO %s.%s values %s",
                        quotename(self.schema),
                        quotename(self._METADATA_TABLE_NAME),
                        str(data)
                       )
            cursor.executemany(sql, data)

    @staticmethod
    def _get_model_id(model):
        """
        Returns the model id (name, version).

        Parameters
        ----------
        model : an instance of SAP HANA PAL or APL model.

        Returns
        -------
        A tuple of two elements (name, version).
        """
        name = getattr(model, 'name', None)
        version = getattr(model, 'version', None)
        return name, version

    @staticmethod
    def _check_valid_name(name):
        """
        Checks the model name is correctly set.
        It must not contain the characters: ',', ';', '"', ''', '\n', '\t'

        Returns
        -------
        If a forbidden character is in the name, raises a ModelStorageError exception.
        """
        forbidden_chars = [';', ',', '"', "'", '\n', '\t']
        if any(c in name for c in forbidden_chars):
            raise ModelStorageError('The model name contains unauthorized characters.')

    def _check_metadata_exists(self):
        """
        Checks if the metadata table exists.

        Returns
        -------
        True/False if yes/no.
        Raise error if database issue.
        """
        with self.connection_context.connection.cursor() as cursor:
            try:
                sql = 'SELECT 1 FROM {schema_name}.{table_name} LIMIT 1'.format(
                    schema_name=quotename(self.schema),
                    table_name=quotename(self._METADATA_TABLE_NAME))
                execute_logged(cursor, sql, self.connection_context.sql_tracer, self.connection_context)
                return True
            except dbapi.Error as err:
                if err.errorcode == 259:
                    return False
                if err.errorcode == 258:
                    raise ModelStorageError('Cannot read the schema. ' + err.errortext)
                raise ModelStorageError('Database issue. ' + err.errortext)
            except pyodbc.Error as err:
                if "(259)" in str(err.args[1]):
                    return False
                if "(258)" in str(err.args[1]):
                    raise ModelStorageError('Cannot read the schema. ' + str(err.args[1]))
                raise ModelStorageError('Database issue. ' + str(err.args[1]))
        return False

    def _delete_metadata(self, name, version):
        """
        Deletes the model metadata.

        Parameters
        ----------
        name : str
            The name of the model to be deleted.
        conn_context : connection_context
            The SAP HANA connection.
        """
        with self.connection_context.connection.cursor() as cursor:
            sql = "DELETE FROM {schema_name}.{table_name} " \
                  "WHERE NAME='{model_name}' " \
                  "AND VERSION={model_version}".format(
                      schema_name=quotename(self.schema),
                      table_name=quotename(self._METADATA_TABLE_NAME),
                      model_name=name.replace("'", "''"),
                      model_version=version)
            execute_logged(cursor, sql, self.connection_context.sql_tracer, self.connection_context)

    def enable_persistent_memory(self, name, version):
        """
        Enable persistent memory.

        Parameters
        ----------
        name : str
            The name of the model.
        version : int
            The model version.
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._enable_persistent_memory(connection_context=self.connection_context,
                                              name=name,
                                              version=version,
                                              js_str=js_str)

    def disable_persistent_memory(self, name, version):
        """
        Disable persistent memory.

        Parameters
        ----------
        name : str
            The name of the model.
        version : int
            The model version.
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._disable_persistent_memory(connection_context=self.connection_context,
                                               name=name,
                                               version=version,
                                               js_str=js_str)

    def load_into_memory(self, name, version):
        """
        Load a model into the memory.

        Parameters
        ----------
        name : str
            The name of the model.
        version : int
            The model version.
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._load_mem(connection_context=self.connection_context,
                              name=name,
                              version=version,
                              js_str=js_str)

    def unload_from_memory(self, name, version, persistent_memory=None):
        """
        Unload a model from the memory. The dataset will be loaded back into memory after next query.

        Parameters
        ----------
        name : str
            The name of the model.
        version : int
            The model version.
        persistent_memory : {'retain', 'delete'}, optional
            Only works when persistent memory is enabled.

            Defaults to None.
        """
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        model_class._unload_mem(
            connection_context=self.connection_context,
            name=name, version=version, js_str=js_str,
            persistent_memory=persistent_memory)

    def set_data_lake_container(self, name):
        """
        Set HDL container name.

        Parameters
        ----------
        name : str
            The name of the HDL container.
        """
        self.data_lake_container = name

    def _delete_model(self, name, version, data_lake_container='SYSRDL#CG'):
        """
        Deletes a model.

        Parameters
        ----------
        name : str
            The name of the model to be deleted.
        version : int
            The model version.
        """
        # Reads the metadata
        metadata = self._get_model_metadata(name, version)
        js_str = metadata['JSON']
        model_class = self._load_class(metadata['CLASS'])
        # Deletes the back-end model by calling the model class static method _delete_model()
        # pylint: disable=protected-access
        model_class._delete_model(
            connection_context=self.connection_context,
            name=name,
            version=version,
            js_str=js_str,
            storage_type=self._get_storge_type(name, version),
            data_lake_container=data_lake_container)
                # Deletes the metadata
        self._delete_metadata(name=name, version=version)

    def _get_model_metadata(self, name, version):
        """
        Reads the json string from the metadata of the model.

        Parameters
        ----------
        name : str
            The model name.
        version : int
            The model version.

        Returns
        -------
        metadata : dict
            A dictionary containing the metadata of the model.
            The keys of the dictionary correspond to the columns of the metadata table:
            {'NAME': 'My Model', 'LIB': 'APL', 'CLASS': ...}
        """
        pd_series = self.list_models(name=name, version=version).head(1)
        return pd_series.iloc[0].to_dict()

    def _get_storge_type(self, name, version):
        """
        Gets the storage type of the model.

        Parameters
        ----------
        name : str
            The model name.
        version : int
            The model version.

        Returns
        -------
        storage_type : str
            - 'default' : HANA default storage.
            - 'date lake' : HANA data lake.

        """
        if self.list_models(name=name, version=version).shape[0] > 0:
            return self.list_models(name=name, version=version).head(1)["STORAGE_TYPE"].iat[0]
        return 'default'

    def _get_new_version_no(self, name):
        """
        Gets the next version number for a model.

        Parameters
        ----------
        name : str
            The model name.

        Returns
        -------
        new_version: int
        """
        last_vers = self._get_last_version_no(name)
        return last_vers + 1

    def _get_last_version_no(self, name):
        """
        Gets the next version number for a model.

        Parameters
        ----------
        name : str
            The model name.

        Returns
        -------
        new_version: int
        """
        cond = "NAME='{}'".format(name.replace("'", "''"))
        sql = "SELECT MAX(version)  NEXT_VER FROM {schema_name}.{table_name}" \
              " WHERE {filter}".format(
                  schema_name=quotename(self.schema),
                  table_name=quotename(self._METADATA_TABLE_NAME),
                  filter=cond)
        hana_df = hana_ml.dataframe.DataFrame(connection_context=self.connection_context, select_statement=sql)
        pd_df = hana_df.collect()
        if pd_df.empty:
            return 0
        new_ver = pd_df['NEXT_VER'].iloc[0]
        if new_ver:  # it could be None
            return new_ver
        return 0

    def set_schedule(self,
                     name,
                     version,
                     schedule_time,
                     connection_userkey,
                     init_params,
                     fit_params,
                     training_dataset_select_statement,
                     storage_type='default',
                     encrypt=None,
                     sslValidateCertificate=None):
        """
        Create the schedule plan.

        Parameters
        ----------
        name : str
            The model name.
        version : int
            The model version.
        schedule_time : {'every x seconds', 'every x minutes', 'every x hours', 'every x weeks'}
            Schedule the training.
        connection_userkey : str
            Userkey generated by HANA hdbuserstore.
        init_params:
            The parameters of the hana_ml object initialization.
        fit_params:
            The parameters of the fit function.
        training_dataset_select_statement:
            The select statement of the training dataset to be scheduled.
        storage_type : {'default', 'HDL'}, optional
            If 'HDL', the model will be saved in HANA Data Lake.
        """
        metadata = self._get_model_metadata(name, version)
        config = json.loads(metadata['SCHEDULE'])
        config['schedule']['schedule_time'] = schedule_time
        config['schedule']['connection']['userkey'] = connection_userkey
        if sslValidateCertificate:
            config['schedule']['connection']['sslValidateCertificate'] = sslValidateCertificate
        if encrypt:
            config['schedule']['connection']['encrypt'] = encrypt
        config['schedule']['hana_ml_obj'] = metadata['CLASS']
        config['schedule']['init_params'] = init_params
        config['schedule']['fit_params'] = fit_params
        config['schedule']['training_dataset_select_statement'] = training_dataset_select_statement
        config['schedule']['storage_type'] = storage_type
        with self.connection_context.connection.cursor() as cursor:
            sql = "UPDATE {}.{} set SCHEDULE = '{}' WHERE NAME = '{}' AND VERSION = {}".format(
                quotename(self.schema),
                quotename(self._METADATA_TABLE_NAME),
                json.dumps(config),
                name,
                version)
            logger.info("Prepare SQL: %s", sql)
            execute_logged(cursor, sql, self.connection_context.sql_tracer, self.connection_context)

    def start_schedule(self, name, version):
        """
        Execute the schedule plan.

        Parameters
        ----------
        name : str
            The model name.
        version : int
            The model version.
        """
        slogger = logging.getLogger("SCHEDULE")
        slogger.setLevel(logging.INFO)
        metadata = self._get_model_metadata(name, version)
        config = json.loads(metadata['SCHEDULE'])
        model_tables = json.loads(metadata['JSON'])['artifacts']['model_tables']
        schema = json.loads(metadata['JSON'])['artifacts']['schema']
        if config["schedule"]["pid"] is not None:
            msg = "The schedule exists and is run by {}. Please terminate it first or create another model!".format(config["schedule"]["client"])
            raise Exception(msg)
        p = Process(target=_task, args=(config, schema, model_tables, self.logfile))
        config["schedule"]["status"] = 'active'
        p.start()
        config["schedule"]["pid"] = p.pid
        config["schedule"]["client"] = socket.gethostname()
        slogger.info("Started the schedule process %s by client %s", config["schedule"]["pid"], config["schedule"]["client"])
        with self.connection_context.connection.cursor() as cursor:
            sql = "UPDATE {}.{} set SCHEDULE = '{}' WHERE NAME = '{}' AND VERSION = {}".format(
                quotename(self.schema),
                quotename(self._METADATA_TABLE_NAME),
                json.dumps(config),
                name,
                version)
            logger.info("Prepare SQL: %s", sql)
            execute_logged(cursor, sql, self.connection_context.sql_tracer, self.connection_context)

    def terminate_schedule(self, name, version):
        """
        Execute the schedule plan.

        Parameters
        ----------
        name : str
            The model name.
        version : int
            The model version.
        """
        slogger = logging.getLogger("SCHEDULE")
        slogger.setLevel(logging.INFO)
        metadata = self._get_model_metadata(name, version)
        config = json.loads(metadata['SCHEDULE'])
        slogger.info("Terminating schedule and killing process %s started by client %s.",
                     config["schedule"]["pid"],
                     config["schedule"]["client"])
        if config["schedule"]["pid"] is not None:
            if config["schedule"]["client"] == socket.gethostname():
                try:
                    os.kill(config["schedule"]["pid"], signal.SIGTERM)
                except OSError as err:
                    slogger.error(err)
                    pass
            else:
                msg = "The schedule exists and is run by {}.".format(config["schedule"]["client"])
                raise Exception(msg)
        config["schedule"]["status"] = 'inactive'
        config["schedule"]["pid"] = None
        config["schedule"]["client"] = None
        with self.connection_context.connection.cursor() as cursor:
            sql = "UPDATE {}.{} set SCHEDULE = '{}' WHERE NAME = '{}' AND VERSION = {}".format(
                quotename(self.schema),
                quotename(self._METADATA_TABLE_NAME),
                json.dumps(config),
                name,
                version)
            logger.info("Prepare SQL: %s", sql)
            execute_logged(cursor, sql, self.connection_context.sql_tracer, self.connection_context)

    def set_logfile(self, loc):
        """
        Set log file location.
        """
        self.logfile = loc

    def upgrade_meta(self):
        """
        Upgrade the meta table to the latest changes.
        """
        meta = self.connection_context.table(self._METADATA_TABLE_NAME, schema=self.schema)
        meta_cols = meta.columns
        if 'SCHEDULE' not in meta_cols and 'STORAGE_TYPE' not in meta_cols:
            meta.save("#TEMP_HANAML_MS_META_TBL")
            self.connection_context.drop_table(self._METADATA_TABLE_NAME, schema=self.schema)
            self.connection_context.create_table(table=self._METADATA_TABLE_NAME,
                                                 table_structure={'NAME' : 'VARCHAR(255)',
                                                                  'VERSION' : 'INT',
                                                                  'LIBRARY' : 'VARCHAR(128)',
                                                                  'CLASS' : 'VARCHAR(255)',
                                                                  'JSON' : 'VARCHAR(5000)',
                                                                  'TIMESTAMP' : 'TIMESTAMP',
                                                                  'STORAGE_TYPE' : 'VARCHAR(255)',
                                                                  'MODEL_STORAGE_VER' : 'INT',
                                                                  'SCHEDULE' : 'VARCHAR(5000)'},
                                                 schema=self.schema)
            storage_type_col = 'STORAGE_TYPE'
            schedule_col = 'SCHEDULE'
            if 'SCHEDULE' not in meta_cols:
                schedule_col = "'{}' SCHEDULE".format(json.dumps(self.schedule_config_template))
            if 'STORAGE_TYPE' not in meta_cols:
                storage_type_col = "'default' STORAGE_TYPE"
            with self.connection_context.connection.cursor() as cur:
                cur.execute("""
                    INSERT INTO {}.{}
                    SELECT NAME,
                        VERSION,
                        LIBRARY,
                        CLASS,
                        JSON,
                        TIMESTAMP,
                        {},
                        MODEL_STORAGE_VER,
                        {}
                    FROM #TEMP_HANAML_MS_META_TBL
                    """.format(quotename(self.schema),
                               quotename(self._METADATA_TABLE_NAME),
                               storage_type_col,
                               schedule_col))
            self.connection_context.drop_table("#TEMP_HANAML_MS_META_TBL")

def _job(config, schema, model_tables, logfile):
    """
    Run the training job.
    """
    slogger = logging.getLogger("SCHEDULE JOB")

    slogger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(filename=logfile, mode='a')
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s")
    fhandler.setFormatter(formatter)
    slogger.addHandler(fhandler)
    hanamlobj_eval = eval(config["schedule"]["hana_ml_obj"] + '(**{})'.format(config["schedule"]["init_params"]))
    slogger.info("Schedule process %s: establishing the HANA connection.", os.getpid())
    conn = ConnectionContext(userkey=config["schedule"]["connection"]["userkey"],
                             encrypt=config["schedule"]["connection"]["encrypt"],
                             sslValidateCertificate=config["schedule"]["connection"]["sslValidateCertificate"])
    slogger.info("Schedule process %s: running the fit function.", os.getpid())
    hanamlobj_eval.fit(DataFrame(conn, config["schedule"]["training_dataset_select_statement"]), **config["schedule"]["fit_params"])
    slogger.info("Schedule process %s: saving the fitted model.", os.getpid())
    if isinstance(model_tables, (list, tuple)):
        for idx, model in enumerate(model_tables):
            hanamlobj_eval.model_[idx].save(where=(schema, model), storage_type=config["schedule"]["storage_type"], force=True)
    else:
        hanamlobj_eval.model_.save(where=(schema, model_tables), storage_type=config["schedule"]["storage_type"], force=True)
    slogger.info("Schedule process %s: closing the HANA connection.", os.getpid())
    conn.close()

def _task(config, schema, model_tables, logfile):
    """
    Schedule the training job.
    """
    val = config["schedule"]["schedule_time"].split(" ")
    eval("schedule.every({0}).{1}.do(_job, config, schema, model_tables, logfile)".format(val[1], val[2]))
    while True:
        schedule.run_pending()
