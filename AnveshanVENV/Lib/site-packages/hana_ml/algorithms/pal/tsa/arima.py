"""
This module contains Python wrapper for PAL ARIMA algorithm.

The following class are available:

    * :class:`ARIMA`
"""
#pylint: disable=too-many-instance-attributes, too-few-public-methods, invalid-name, too-many-statements
#pylint: disable=too-many-lines, line-too-long, too-many-arguments, too-many-branches, attribute-defined-outside-init
#pylint: disable=simplifiable-if-statement, too-many-locals, bare-except
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
from hana_ml.dataframe import quotename
#from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.algorithms.pal.utility import check_pal_function_exist
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    TupleOfIntegers,
    pal_param_register,
    try_drop,
    require_pal_usable,
    execute_logged
)

logger = logging.getLogger(__name__)

def _convert_index_from_timestamp_to_int(data, key=None):
    if key is None:
        if data.index is None:
            key = data.columns[0]
        else:
            key = data.index
    return data.add_id(key + '(INT)', ref_col=key).deselect(key)

def _is_index_int(data, key=None):
    if key is None:
        if data.index is None:
            key = data.columns[0]
        else:
            key = data.index
    try:
        if not 'INT' in data.get_table_structure()[key].upper():
            return False
    except Exception as err:
        logger.warning(err)
        pass
    return True

def _get_forecast_starttime_and_timedelta(data, key=None, is_index_int=True):
    if key is None:
        if data.index is None:
            key = data.columns[0]
        else:
            key = data.index
    max_ = data.select(key).max()
    min_ = data.select(key).min()
    delta = (max_ - min_) / (data.count() - 1)
    if is_index_int:
        return max_ + delta, delta
    return max_ + delta, delta.total_seconds()

class _ARIMABase(PALBase):
    method_map = {'css':0, 'mle':1, 'css-mle':2}
    forecast_method_map = {'formula_forecast':0, 'innovations_algorithm':1}
    forecast_method_predict_map = {'formula_forecast':0, 'innovations_algorithm':1, 'truncation_algorithm':2}

    def __init__(self,
                 order=None,
                 seasonal_order=None,
                 method='css-mle',
                 include_mean=None,
                 forecast_method=None,
                 output_fitted=True,
                 thread_ratio=None,
                 background_size=None):


        if not hasattr(self, 'hanaml_parameters'):
            setattr(self, 'hanaml_parameters', pal_param_register())
        super(_ARIMABase, self).__init__()
        #P, D, Q in PAL has been combined to be one parameter `order`
        self.order = self._arg('order', order, TupleOfIntegers)
        if self.order is not None and len(self.order) != 3:
            msg = ('order must contain exactly 3 integers for regression order, ' +
                   'differentiation order and moving average order!')
            logger.error(msg)
            raise ValueError(msg)
        #seasonal P, D, Q and seasonal period in PAL has been combined
        #to be one parameter `seasonal order`
        self.seasonal_order = self._arg('seasonal_order', seasonal_order, TupleOfIntegers)
        if self.seasonal_order is not None and len(self.seasonal_order) != 4:
            msg = ('seasonal_order must contain exactly 4 integers for regression order, ' +
                   'differentiation order, moving average order for seasonal part' +
                   'and seasonal period.')
            logger.error(msg)
            raise ValueError(msg)
        if (self.seasonal_order is not None and
                any(s_order > 0 for s_order in self.seasonal_order[:3]) and
                self.seasonal_order[3] <= 1):
            msg = ('seasonal_period must be larger than 1.')
            logger.error(msg)
            raise ValueError(msg)
        self.method = self._arg('method', method, self.method_map)
        self.forecast_method = self._arg('forecast_method',
                                         forecast_method,
                                         self.forecast_method_map)
        self.output_fitted = self._arg('output_fitted', output_fitted, bool)
        self.include_mean = self._arg('include_mean', include_mean, bool)
        if (self.order is not None and
                self.seasonal_order is not None and
                self.order[1] + self.seasonal_order[1] > 1 and
                self.include_mean is not None):
            msg = ('include_mean is only valid when the sum of differentiation order ' +
                   'seasonal_period is not larger than 1.')
            logger.error(msg)
            raise ValueError(msg)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.background_size = self._arg('background_size', background_size, int)

        self.forecast_start = None
        self.timedelta = None
        self.is_index_int = None

    @trace_sql
    def _fit(self, data, endog):
        if not hasattr(self, 'hanaml_fit_params'):
            setattr(self, 'hanaml_fit_params', pal_param_register())
        conn = data.connection_context
        require_pal_usable(conn)
        self.conn_context = conn
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['MODEL', 'FIT']
        outputs = ['#PAL_ARIMA_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        model_tbl, fit_tbl = outputs

        param_rows = [
            ('P', None if self.order is None else self.order[0], None, None),
            ('D', None if self.order is None else self.order[1], None, None),
            ('Q', None if self.order is None else self.order[2], None, None),
            ('SEASONAL_P',
             None if self.seasonal_order is None else self.seasonal_order[0],
             None, None),
            ('SEASONAL_D',
             None if self.seasonal_order is None else self.seasonal_order[1],
             None, None),
            ('SEASONAL_Q',
             None if self.seasonal_order is None else self.seasonal_order[2],
             None, None),
            ('SEASONAL_PERIOD',
             None if self.seasonal_order is None else self.seasonal_order[3],
             None, None),
            ('METHOD', self.method, None, None),
            ('INCLUDE_MEAN', self.include_mean, None, None),
            ('FORECAST_METHOD', self.forecast_method, None, None),
            ('OUTPUT_FITTED', self.output_fitted, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('DEPENDENT_VARIABLE', None, None, endog),
            ('BACKGROUND_SIZE', self.background_size, None, None)
            ]

        data_ = data
        try:
            self._call_pal_auto(conn,
                                'PAL_ARIMA',
                                data_,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, outputs)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, outputs)
            raise
        self.model_ = conn.table(model_tbl)
        self.fitted_ = (conn.table(fit_tbl) if self.output_fitted
                        else None)
        self.explainer_ = None # only has content when predict with show_explainer = True is invoked

    def set_conn(self, connection_context):
        """
        Set connection context for an ARIMA instance.

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
    def _predict(self,
                 data,
                 forecast_method,
                 forecast_length,
                 show_explainer,
                 thread_ratio,
                 top_k_attributions,
                 trend_mod,
                 trend_width,
                 seasonal_width):
        """
        Makes time series forecast based on the estimated ARIMA model.
        """

        conn = self.conn_context
        #if self.background_size is not None and show_explainer is not True:

        if show_explainer is not True:
            param_rows = [
                ("FORECAST_METHOD", forecast_method, None, None),
                ("FORECAST_LENGTH", forecast_length, None, None)]
        else:
            param_rows = [
                ("FORECAST_METHOD", forecast_method, None, None),
                ("FORECAST_LENGTH", forecast_length, None, None),
                ("THREAD_RATIO", None, thread_ratio, None),
                ("TOP_K_ATTRIBUTIONS", top_k_attributions, None, None),
                ("TREND_MOD", None, trend_mod, None),
                ("TREND_WIDTH", None, trend_width, None),
                ("SEASONAL_WIDTH", None, seasonal_width, None)]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_ARIMA_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)
        decompose_tbl = "#PAL_ARIMA_FORECAST_DECOMPOSITION_TBL_{}_{}".format(self.id, unique_id)
        data_tbl = None

        try:
            if data is None:
                data_tbl = "#PAL_ARIMA_FORECAST_DATA_TBL_{}_{}".format(self.id, unique_id)
                with conn.connection.cursor() as cur:
                    execute_logged(cur,
                                   'CREATE LOCAL TEMPORARY COLUMN TABLE {} ("TIMESTAMP" INTEGER,"SERIES" DOUBLE)'.format(data_tbl),
                                   conn.sql_tracer,
                                   conn)
                    data = conn.table(data_tbl)
                if not conn.connection.getautocommit():
                    conn.connection.commit()
            if show_explainer is not True:
                self._call_pal_auto(conn,
                                    'PAL_ARIMA_FORECAST',
                                    data,
                                    self.model_,
                                    ParameterTable().with_data(param_rows),
                                    result_tbl)
            else:
                if check_pal_function_exist(conn, 'ARIMAEXPLAIN%', like=True):
                    self._call_pal_auto(conn,
                                        'PAL_ARIMA_EXPLAIN',
                                        data,
                                        self.model_,
                                        ParameterTable().with_data(param_rows),
                                        result_tbl,
                                        decompose_tbl)
                else:
                    msg = 'The version of SAP HANA does not support ARIMA explainer. Please set show_explainer=False!'
                    logger.error(msg)
                    raise ValueError(msg)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            if data_tbl is not None:
                try_drop(conn, data_tbl)
            try_drop(conn, result_tbl)
            if show_explainer is True:
                try_drop(conn, decompose_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            if data_tbl is not None:
                try_drop(conn, data_tbl)
            try_drop(conn, result_tbl)
            if show_explainer is True:
                try_drop(conn, decompose_tbl)
            raise
        if show_explainer is True:
            self.explainer_ = conn.table(decompose_tbl)

        return conn.table(result_tbl)

class ARIMA(_ARIMABase):
    r"""
    Autoregressive Integrated Moving Average ARIMA(p, d, q) model.

    .. note::
        PAL ARIMA algorithm contains four functions: ARIMA, SARIMA, ARIMAX and SARIMA depending on \n
        whether seasonal information and external (intervention) data are provided.

    Parameters
    ----------

    order : (p, d, q), tuple of int, optional
        - p: value of the auto regression order.
        - d: value of the differentiation order.
        - q: value of the moving average order.

        Defaults to (0, 0, 0).

    seasonal_order : (P, D, Q, s), tuple of int, optional
        - P: value of the auto regression order for the seasonal part.
        - D: value of the differentiation order for the seasonal part.
        - Q: value of the moving average order for the seasonal part.
        - s: value of the seasonal period.

        Defaults to (0, 0, 0, 0).

    method : {'css', 'mle', 'css-mle'}, optional
        - 'css': use the conditional sum of squares.
        - 'mle': use the maximized likelihood estimation.
        - 'css-mle': use css to approximate starting values first and then mle to fit.

        Defaults to 'css-mle'.

    include_mean : bool, optional
        ARIMA model includes a constant part if True.
        Valid only when d + D <= 1 (d is defined in ``order`` and D is defined in ``seasonal_order``).

        Defaults to True if d + D = 0 else False.

    forecast_method : {'formula_forecast', 'innovations_algorithm'}, optional

        - 'formula_forecast': compute future series via formula.
        - 'innovations_algorithm': apply innovations algorithm to compute future series,
          which requires more original information to be stored.

        Store information for the subsequent forecast method.

        Defaults to 'innovations_algorithm'.

    output_fitted : bool, optional
        Output fitted result and residuals if True.

        Defaults to True.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The ratio of available threads.

            - 0: single thread.
            - 0~1: percentage.
            - Others: heuristically determined.

        Defaults to -1.

    background_size : int, optional
        Indicates the nubmer of data points used in ARIMA with explainations in the predict function.
        If you want to use the ARIMA with explainations, you must set ``background_size`` to be a positive value or -1(auto mode)
        when initializing an ARIMA instance the and then set ``show_explainer=True`` in the predict function.

        Defaults to NULL(no explainations).

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    fitted_ : DateFrame

        Fitted values and residuals.

    explainer_ : DataFrame

        The with explainations with decomposition of trend, seasonal, transitory, irregular
        and reason code of exogenous variables.
        The attribute only appear when setting ``background_size`` when initializing an ARIMA instance
        and ``show_explainer=True`` in the predict function.

    Examples
    --------

    ARIMA example:

    Input dataframe df for ARIMA:

    >>> df.head(5).collect()
        TIMESTAMP    Y
        1            -0.636126431
        2             3.092508651
        3            -0.73733556
        4            -3.142190983
        5             2.088819813

    Create an ARIMA instance:

    >>> arima = ARIMA(order=(0, 0, 1), seasonal_order=(1, 0, 0, 4), method='mle', thread_ratio=1.0)

    Perform fit on the given data:

    >>> arima.fit(data=df)

    Show the output:

    >>> arima.head(5).model_.collect()
         KEY    VALUE
    0    p      0
    1    AR
    2    d      0
    3    q      1
    4    MA    -0.141073

    >>> arima.fitted_.head(3).collect().set_index('TIMESTAMP')
         TIMESTAMP   FITTED      RESIDUALS
    1    1           0.023374   -0.659500
    2    2           0.114596    2.977913
    3    3          -0.396567   -0.340769

    Perform predict on the model:

    >>> result = arima.predict(forecast_method='innovations_algorithm', forecast_length=10)

    Show the output:

    >>> result.head(3).collect()
        TIMESTAMP   FORECAST     SE          LO80         HI80        LO95        HI95
    0   0           1.557783     1.302436   -0.111357    3.226922    -0.994945    4.110511
    1   1           3.765987     1.315333    2.080320    5.451654     1.187983    6.343992
    2   2          -0.565599     1.315333   -2.251266    1.120068    -3.143603    2.012406

    If you want to see the decomposed result of predict result, you could set ``background_size``
    when initializing an ARIMA instance and set ``show_explainer`` = True in the predict():

    >>> arima = ARIMA(order=(0, 0, 1),
                      seasonal_order=(1, 0, 0, 4),
                      method='mle',
                      thread_ratio=1.0,
                      background_size = 10)
    >>> result = arima.predict(forecast_method='innovations_algorithm',
                               forecast_length=3,
                               allow_new_index=False,
                               show_explainer=True)

    Show the explainer\_ of arima instance:

    >>> arima.explainer_.head(3).collect()
       ID     TREND SEASONAL TRANSITORY IRREGULAR                                          EXOGENOUS
    0   0  1.179043     None       None      None  [{"attr":"X","val":-0.49871412549199997,"pct":...
    1   1  1.252138     None       None      None  [{"attr":"X","val":-0.27390052549199997,"pct":...
    2   2  1.362164     None       None      None  [{"attr":"X","val":-0.19046313238292013,"pct":...

    ARIMAX example:

    Input dataframe df for ARIMAX:

    >>> df.head(5).collect()
       ID    Y                   X
       1     1.2                 0.8
       2     1.34845613096197    1.2
       3     1.32261090809898    1.34845613096197
       4     1.38095306748554    1.32261090809898
       5     1.54066648969168    1.38095306748554

    Create an ARIMAX instance:

    >>> arimax = ARIMA(order=(1, 0, 1), method='mle', thread_ratio=1.0)

    Perform fit on the given data:

    >>> arimax.fit(data=df, endog='Y')

    Show the output:

    >>> arimax.model_.collect().head(5)
         KEY    VALUE
         p      1
         AR     0.302207
         d      0
         q      1
         MA     0.291575

    >>> arimax.fitted_.head(3).collect().set_index('TIMESTAMP')
       TIMESTAMP   FITTED     RESIDUALS
       1           1.182363    0.017637
       2           1.416213   -0.067757
       3           1.453572   -0.130961

    Perform predict on the ARIMAX model:

    Input dataframe df2 for ARIMAX predict:

    >>> df2.head(5).collect()
        TIMESTAMP    X
        1            0.800000
        2            1.200000
        3            1.348456
        4            1.322611
        5            1.380953

    >>> result = arimax.predict(df2, forecast_method='innovations_algorithm', forecast_length=5)

    Show the output:

    >>> result.head(3).collect()
        TIMESTAMP    FORECAST    SE          LO80         HI80        LO95        HI95
        0            1.195952    0.093510    1.076114     1.315791    1.012675    1.379229
        1            1.411284    0.108753    1.271912     1.550657    1.198132    1.624436
        2            1.491856    0.110040    1.350835     1.632878    1.276182    1.707530

    """
    def fit(self, data, key=None, endog=None, exog=None):
        """
        Generates ARIMA models with given parameters.

        Parameters
        ----------

        data : DataFrame

            Input data which at least have two columns: key and endog.

            We also support ARIMAX which needs external data (exogenous variables).

        key : str, optional

            The timestamp column of data. The type of key column should be INTEGER,
            TIMESTAMP, DATE or SECONDDATE.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional

            The endogenous variable, i.e. time series. The type of endog column could be
            INTEGER, DOUBLE or DECIMAL(p,s).

            Defaults to the first non-key column of data if not provided.

        exog : str or a list of str, optional

            An optional array of exogenous variables. The type of exog column could be
            INTEGER, DOUBLE or DECIMAL(p,s).

            Valid only for ARIMAX.

            Defaults to None. Please set this parameter explicitly if you have exogenous variables.

        Returns
        -------
        A fitted object of class "ARIMA".
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        if data is None:
            msg = ('The data for fit cannot be None!')
            logger.error(msg)
            raise ValueError(msg)

        # validate key, endog, exog
        cols = data.columns
        index = data.index
        key = self._arg('key', key, str)

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

        if key is not None and key not in cols:
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)
        cols.remove(key)

        endog = self._arg('endog', endog, str)
        if endog is not None:
            if endog not in cols:
                msg = ('Please select endog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            endog = cols[0]
        cols.remove(endog)

        if exog is not None:
            if isinstance(exog, str):
                exog = [exog]
            exog = self._arg('exog', exog, ListOfStrings)
            if set(exog).issubset(set(cols)) is False:
                msg = ('Please select exog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            exog = []

        data_ = data[[key] + [endog] + exog]
        self.fit_data = data_

        self.is_index_int = _is_index_int(data_, key)
        if not self.is_index_int:
            super(ARIMA, self)._fit(_convert_index_from_timestamp_to_int(data_, key), endog)
        else:
            super(ARIMA, self)._fit(data_, endog)
        try:
            self.forecast_start, self.timedelta = _get_forecast_starttime_and_timedelta(data_, key, self.is_index_int)
        except Exception as err:
            logger.warning(err)
            pass
        return self

    def predict(self,
                data=None,
                key=None,
                forecast_method=None,
                forecast_length=None,
                allow_new_index=False,
                show_explainer=False,
                thread_ratio=None,
                top_k_attributions=None,
                trend_mod=None,
                trend_width=None,
                seasonal_width=None):
        r"""
        Makes time series forecast based on the estimated ARIMA model.

        Parameters
        ----------

        data : DataFrame, optional

            Index and exogenous variables for forecast. For ARIMAX only.

            Defaults to None.

        key : str, optional

            The timestamp column of data. The data type of key column should be
            INTEGER, TIMESTAMP, DATE or SECONDDATE. For ARIMAX only.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        forecast_method : {'formula_forecast', 'innovations_algorithm', 'truncation_algorithm'}, optional
            Specify the forecast method.

            If 'forecast_method' in fit function is 'formula_forecast', no enough information is stored, so the formula method is adopted instead.

            Truncation algorithm is much faster than innovations algorithm when the AR representation of
            ARIMA model can be truncated to finite order.

            - 'formula_forecast': forecast via formula.
            - 'innovations_algorithm': apply innovations algorithm to forecast.
            - 'truncation_algorithm'.

            Defaults to 'innovations_algorithm'.

        forecast_length : int, optional

            Number of points to forecast.

            Valid only when ``data`` is None.

            In ARIMAX, the forecast length is the same as the length of the input data.

            Defaults to None.

        allow_new_index : bool, optional

            Indicate whether a new index column is allowed in the result.
            - True: return the result with new integer or timestamp index column.
            - False: return the result with index column starting from 0.

            Defaults to False.

        show_explainer : bool, optional
            Indicate whether to invoke the ARIMA with explainations function in the predict.
            Only valide when ``background_size`` is set when initializing an ARIMA instance.

            If true, the contributions of trend, seasonal, transitory irregular and exogenous are
            shown in a attribute called explainer\_ of arima/auto arima instance.

            Defaults to False.

        thread_ratio : float, optional
            Controls the proportion of available threads to use.
            The ratio of available threads.

                - 0: single thread
                - 0~1: percentage
                - Others: heuristically determined

        Defaults to -1. Valid only when ``show_explainer`` is True.

        top_k_attributions : int, optional

            Specifies the number of attributes with the largest contribution that will be output.
            0-contributed attributes will not be output
            Valid only when ``show_explainer`` is True.

            Defaults to 10.

        trend_mod : double, optional

            The real AR roots with inverse modulus larger than TREND_MOD will be integrated into trend component.
            Valid only when ``show_explainer`` is True.
            Cannot be smaller than 0.

            Defaults to 0.4.

        trend_width : double, optional

            Specifies the bandwidth of spectrum of trend component in unit of rad.
            Valid only when ``show_explainer`` is True. Cannot be smaller than 0.

            Defaults to 0.035.

        seasonal_width : double, optional

            Specifies the bandwidth of spectrum of seasonal component in unit of rad.
            Valid only when ``show_explainer`` is True. Cannot be smaller than 0.

            Defaults to 0.035.

        Returns
        -------

        DataFrame
            Forecasted values, structured as follows:

              - ID, type INTEGER or TIMESTAMP.
              - FORECAST, type DOUBLE, forecast value.
              - SE, type DOUBLE, standard error.
              - LO80, type DOUBLE, low 80% value.
              - HI80, type DOUBLE, high 80% value.
              - LO95, type DOUBLE, low 95% value.
              - HI95, type DOUBLE, high 95% value.

        """
        if getattr(self, 'model_', None) is None:
            msg = 'Model is not initialized. Perform a fit first!'
            logger.error(msg)
            raise ValueError(msg)

        if (self.forecast_method == 0) and (forecast_method == 'innovations_algorithm' or forecast_method is None):
            msg = ('No enough information is stored for innovations_algorithm, ' +
                   'please use formula_forecast instead!')
            logger.error(msg)
            raise ValueError(msg)

        if ((self.background_size is None) or (self.background_size == 0)) and (show_explainer is True):
            msg = 'Please set the value of "background_size" to obtain the ARIMA explaination!'
            logger.error(msg)
            raise ValueError(msg)

        forecast_method = self._arg('forecast_method',
                                    forecast_method,
                                    self.forecast_method_predict_map)
        forecast_length = self._arg('forecast_length', forecast_length, int)

        if show_explainer is True:
            thread_ratio = self._arg('thread_ratio', thread_ratio, float)
            top_k_attributions = self._arg('top_k_attributions', top_k_attributions, int)
            trend_mod = self._arg('trend_mod', trend_mod, float)
            trend_width = self._arg('trend_width', trend_width, float)
            seasonal_width = self._arg('seasonal_width', seasonal_width, float)

        # validate key
        key = self._arg('key', key, str)

        if ((key is not None) and (data is not None) and (key not in data.columns)):
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)

        data_ = data

        # prepare the data, which could be empty or combination of key(must be the first column) and external data.
        if data is None:
            self.predict_data = None
            unique_id = str(uuid.uuid1()).replace('-', '_').upper()
            data_name = "#PAL_ARIMA_PREDICT_DATA_{}".format(unique_id)
            with self.conn_context.connection.cursor() as cur:
                execute_logged(cur,
                               'CREATE LOCAL TEMPORARY  COLUMN TABLE {} ("TIMESTAMP" INTEGER, "Y" DOUBLE)'.format(data_name),
                               self.conn_context.sql_tracer,
                               self.conn_context)
            if not self.conn_context.connection.getautocommit():
                self.conn_context.connection.commit()
            data_ = self.conn_context.table(data_name)
            # for default key value
            if key is None:
                key = 'TIMESTAMP'
        else:
            index = data.index
            cols = data.columns
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
            exog = cols
            data_ = data[[key] + exog]
            self.predict_data = data_
            is_index_int = _is_index_int(data_, key)
            key_output = key
            if not is_index_int:
                data_ = _convert_index_from_timestamp_to_int(data_, key)
                key = key_output

        result = super(ARIMA, self)._predict(data_,
                                             forecast_method,
                                             forecast_length,
                                             show_explainer,
                                             thread_ratio,
                                             top_k_attributions,
                                             trend_mod,
                                             trend_width,
                                             seasonal_width)
        if not allow_new_index:
            return result

        # Note that if model_storage is used, self.in_index_int would be None, so allow_new_index is useless.
        if self.is_index_int:
            return self.conn_context.sql("""
                                        SELECT {0} + {1} AS {9},
                                        {3},
                                        {4},
                                        {5},
                                        {6},
                                        {7},
                                        {8}
                                        FROM ({2})
                                        """.format(self.forecast_start,
                                                   quotename(result.columns[0]),
                                                   result.select_statement,
                                                   quotename(result.columns[1]),
                                                   quotename(result.columns[2]),
                                                   quotename(result.columns[3]),
                                                   quotename(result.columns[4]),
                                                   quotename(result.columns[5]),
                                                   quotename(result.columns[6]),
                                                   quotename(key)))
        return self.conn_context.sql("""
                                    SELECT ADD_SECONDS('{0}', {1} * {2}) AS {10},
                                    {4},
                                    {5},
                                    {6},
                                    {7},
                                    {8},
                                    {9}
                                    FROM ({3})
                                    """.format(self.forecast_start,
                                               quotename(result.columns[0]),
                                               self.timedelta,
                                               result.select_statement,
                                               quotename(result.columns[1]),
                                               quotename(result.columns[2]),
                                               quotename(result.columns[3]),
                                               quotename(result.columns[4]),
                                               quotename(result.columns[5]),
                                               quotename(result.columns[6]),
                                               quotename(key)))
