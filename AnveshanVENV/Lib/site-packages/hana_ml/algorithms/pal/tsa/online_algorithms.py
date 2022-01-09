"""
This module contains Python wrapper for PAL online time series algorithms.

The following class are available:

    * :class:`OnlineARIMA`
"""
#pylint: disable=too-many-lines, line-too-long, too-many-instance-attributes, too-many-locals, too-many-statements, too-many-branches,  bare-except
#pylint: disable=consider-using-f-string
#pylint: disable=broad-except
import logging
import uuid
import warnings
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.dataframe import quotename
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    TupleOfIntegers,
    pal_param_register,
    try_drop,
    require_pal_usable,
    execute_logged
)
from hana_ml.algorithms.pal.tsa.arima import _convert_index_from_timestamp_to_int, _is_index_int
from hana_ml.algorithms.pal.tsa.arima import _get_forecast_starttime_and_timedelta
logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class OnlineARIMA(PALBase):
    r"""
    Online Autoregressive Integrated Moving Average ARIMA(p, d, q) model.

    Parameters
    ----------

    order : (p, d, m), tuple of int, optional
        - p: value of the auto regression order.
        - d: value of the differentiation order.
        - m: extended order needed to transform all moving-averaging term to AR term.

        Defaults to (1, 0, 0).

    learning_rate : float, optional
        Learning rate. Must be greater than zero.

        Calculated according to order(p, d, m).

    epsilon : float, optional
        Convergence criterion.

        Calculated according to learning_rate, p and m in the order.

    output_fitted : bool, optional
        Output fitted result and residuals if True.
        Defaults to True.

    random_state : int, optional
        Specifies the seed for random number generator.

        - 0: use the current time (in second) as seed
        - Others: use the specified value as seed.

        Default to 0.

    random_initialization : bool, optional
        Whether randomly generate initial state

        - False: set all to zero
        - True: use random values

        Defaults to False.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    fitted_ : DateFrame

        Fitted values and residuals.

    """
    def __init__(self,#pylint: disable=too-many-arguments
                 order=None,
                 learning_rate=None,
                 epsilon=None,
                 output_fitted=True,
                 random_state=None,
                 random_initialization=None):

        setattr(self, 'hanaml_parameters', pal_param_register())
        super(OnlineARIMA, self).__init__()
        #P, D, Q in PAL has been combined to be one parameter `order`
        self.order = self._arg('order', order, TupleOfIntegers)
        if self.order is not None and len(self.order) != 3:
            msg = ('order must contain exactly 3 integers for p, d, m respectively!')
            logger.error(msg)
            raise ValueError(msg)

        self.learning_rate = self._arg('learning_rate', learning_rate, float)
        self.epsilon = self._arg('epsilon', epsilon, float)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.random_state = self._arg('random_state', random_state, int)
        self.random_initialization = self._arg('random_initialization', random_initialization, bool)
        self.model_ = None
        self.forecast_start = None
        self.timedelta = None
        self.is_index_int = None

    def set_conn(self, connection_context):
        """
        Set connection context for OnlineARIMA instance.

        Parameters
        ----------
        connection_context : ConnectionContext
            The connection to the SAP HANA system.

        Returns
        -------
        None.

        """
        self.conn_context = connection_context

    @trace_sql
    def partial_fit(self,
                    data,
                    key=None,
                    endog=None,
                    **kwargs):
        """
        Generates ARIMA models with given orders.

        Parameters
        ----------

        data : DataFrame

            DataFrame containing the data.

        key : str, optional

            The timestamp column of data. The type of key column is int.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional

            The endogenous variable, i.e. time series.

            Defaults to the first non-key column of data if not provided.

        **kwargs : keyword arguments, optional

            The value of ``learning_rate`` and ``epsilon`` could be reset in the model.

            For example, assume we have a OnlineARIMA object `oa` and we want to reset the value of ``learning_rate`` in the new training.
                >>> oa.partial_fit(new_data, learning_rate=0.02)

        Returns
        -------
        A fitted object of class "OnlineARIMA".

        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        if data is None:
            msg = ('The data for fit cannot be None!')
            logger.error(msg)
            raise ValueError(msg)

        # validate key, endog, exog, kwargs
        cols = data.columns
        key = self._arg('key', key, str)
        if key is not None and key not in cols:
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)

        index = data.index
        if index is not None:
            if key is None:
                if not isinstance(index, str):
                    key = cols[0]
                    warn_msg = "The index of data is not a single column and key is None, so the first column of data is used as key!"
                    warnings.warn(message=warn_msg)
                else:
                    key = index
            else:
                if key != index:
                    warn_msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                    "and the designated index column '{}'.".format(index)
                    warnings.warn(message=warn_msg)
        else:
            if key is None:
                key = cols[0]
        cols.remove(key)

        endog = self._arg('endog', endog, str)
        if endog is not None:
            if endog not in cols:
                msg = ('Please select endog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            endog = cols[0]

        data_ = data[[key] + [endog]]

        self.is_index_int = _is_index_int(data_, key)
        try:
            self.forecast_start, self.timedelta = _get_forecast_starttime_and_timedelta(data_, key, self.is_index_int)
        except Exception as err:
            logger.warning(err)
            pass

        if not self.is_index_int:
            data_ = _convert_index_from_timestamp_to_int(data_, key)

        conn = data.connection_context
        require_pal_usable(conn)
        self.conn_context = conn

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        input_model_tbl = "#PAL_ONLINE_ARIMA_MODEL_IN_TBL_{}_{}".format(self.id, unique_id)

        try:
            with conn.connection.cursor() as cur:
                execute_logged(cur,
                               'CREATE LOCAL TEMPORARY COLUMN TABLE {} ("KEY" NVARCHAR(100), "VALUE" NVARCHAR(5000))'.format(input_model_tbl),
                               conn.sql_tracer,
                               conn)
                if self.model_ is not None:
                    execute_logged(cur,
                                   'INSERT INTO {} SELECT * FROM ({})'.format(input_model_tbl, self.model_.select_statement),
                                   conn.sql_tracer,
                                   conn)
                    if kwargs:
                        if 'learning_rate' in kwargs:
                            if self.model_ is not None:
                                learning_rate_ = self._arg('learning_rate', kwargs['learning_rate'], float)
                                with conn.connection.cursor() as cursor:
                                    execute_logged(cursor,
                                                   "UPDATE {} SET VALUE='{}' WHERE KEY='lrate'".format(input_model_tbl,
                                                                                                       learning_rate_),
                                                   conn.sql_tracer,
                                                   conn)
                        if 'epsilon' in kwargs:
                            if self.model_ is not None:
                                epsilon_ = self._arg('epsilon', kwargs['epsilon'], float)
                                with conn.connection.cursor() as cursor:
                                    execute_logged(cursor,
                                                   "UPDATE {} SET VALUE='{}' WHERE KEY='epsilon'".format(input_model_tbl,
                                                                                                         epsilon_),
                                                   conn.sql_tracer,
                                                   conn)
            conn.connection.commit()
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
        input_model = conn.table(input_model_tbl)
        outputs = ['MODEL', 'FIT']
        outputs = ['#PAL_ONLINE_ARIMA_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        model_tbl, fit_tbl = outputs

        param_rows = [
            ('P', None if self.order is None else self.order[0], None, None),
            ('D', None if self.order is None else self.order[1], None, None),
            ('M', None if self.order is None else self.order[2], None, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('LEARNING_RATE', None, self.learning_rate, None),
            ('EPSILON', None, self.epsilon, None),
            ('SEED', self.random_state, None, None),
            ('RANDOM_INITIALIZATION', self.random_initialization, None, None)
            ]

        try:
            self._call_pal_auto(conn,
                                'PAL_ONLINE_ARIMA',
                                data_,
                                ParameterTable().with_data(param_rows),
                                input_model,
                                *outputs)
            try_drop(conn, input_model_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, input_model_tbl)
            try_drop(conn, outputs)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, input_model_tbl)
            try_drop(conn, outputs)
            raise
        #pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)
        self.fitted_ = (conn.table(fit_tbl) if self.output_fitted else None)

        return self

    @trace_sql
    def predict(self, forecast_length=None, allow_new_index=False):
        """
        Makes time series forecast based on the estimated online ARIMA model.

        Parameters
        ----------
        forecast_length : int, optional
            Forecast horizon, i.e. number of future points to forecast.

            Defaults to 1.

        allow_new_index : bool, optional

            Indicate whether a new index column is allowed in the result.
            - True: return the result with new integer or timestamp index column.
            - False: return the result with index column starting from 0.

            Defaults to False.

        Returns
        -------
        DataFrame
            Prediction result, i.e. forecasted values within specified horizon, structured as follows:
                - 1st column : timestamp
                - 2nd column : forecast value
        """
        conn = self.conn_context
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_ONLINE_ARIMA_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)

        param_rows = [
            ("FORECAST_LENGTH", forecast_length, None, None)]

        try:
            self._call_pal_auto(conn,
                                'PAL_ONLINE_ARIMA_FORECAST',
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, result_tbl)
            raise
        result = conn.table(result_tbl)
        if not allow_new_index:
            return result
        if self.is_index_int:
            return self.conn_context.sql("""
                                        SELECT {0} + {1} AS {1},
                                        {3}
                                        FROM ({2})
                                        """.format(self.forecast_start,
                                                   quotename(result.columns[0]),
                                                   result.select_statement,
                                                   quotename(result.columns[1])))
        return self.conn_context.sql("""
                                    SELECT ADD_SECONDS('{0}', {1} * {2}) AS {1},
                                    {4}
                                    FROM ({3})
                                    """.format(self.forecast_start,
                                               quotename(result.columns[0]),
                                               self.timedelta,
                                               result.select_statement,
                                               quotename(result.columns[1])))
