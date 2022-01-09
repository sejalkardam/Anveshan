#pylint: disable=too-many-lines, line-too-long, invalid-name, relative-beyond-top-level, too-few-public-methods
#pylint: disable=too-many-arguments, too-many-instance-attributes, too-few-public-methods, too-many-branches
#pylint: disable=attribute-defined-outside-init, too-many-statements, too-many-locals, bare-except
#pylint: disable=unused-import, super-with-arguments, c-extension-no-member
#pylint: disable=broad-except
"""
This module contains Python wrappers for PAL exponential smoothing algorithms.

The following classes are available:

    * :class:`SingleExponentialSmoothing`
    * :class:`DoubleExponentialSmoothing`
    * :class:`TripleExponentialSmoothing`
    * :class:`AutoExponentialSmoothing`
    * :class:`BrownExponentialSmoothing`
    * :class:`Croston`

"""
import logging
import uuid
import warnings
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.dataframe import quotename
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfTuples,
    try_drop,
    require_pal_usable
)
from hana_ml.algorithms.pal.unified_exponentialsmoothing import UnifiedExponentialSmoothing
from hana_ml.algorithms.pal.tsa.arima import _convert_index_from_timestamp_to_int, _is_index_int
from hana_ml.algorithms.pal.tsa.arima import _get_forecast_starttime_and_timedelta
from hana_ml.algorithms.pal.utility import check_pal_function_exist
logger = logging.getLogger(__name__)

class _ExponentialSmoothingBase(PALBase):

    trend_test_map = {'mk': 1, 'difference-sign': 2}
    def __init__(self,
                 model_selection=None,# Auto ESM
                 forecast_model_name=None,# Auto ESM
                 optimizer_time_budget=None,# Auto ESM
                 max_iter=None,# Auto ESM
                 optimizer_random_seed=None,# Auto ESM
                 thread_ratio=None,# Auto ESM
                 alpha=None,
                 beta=None,
                 gamma=None,
                 phi=None,
                 forecast_num=None,
                 seasonal_period=None,
                 seasonal=None,
                 initial_method=None,
                 training_ratio=None,
                 damped=None,
                 accuracy_measure=None,
                 seasonality_criterion=None,# Auto ESM
                 trend_test_method=None,# Auto ESM
                 trend_test_alpha=None,# Auto ESM
                 alpha_min=None, # Auto ESM
                 beta_min=None,# Auto ESM
                 gamma_min=None,# Auto ESM
                 phi_min=None,# Auto ESM
                 alpha_max=None,# Auto ESM
                 beta_max=None,# Auto ESM
                 gamma_max=None,# Auto ESM
                 phi_max=None,# Auto ESM
                 prediction_confidence_1=None,
                 prediction_confidence_2=None,
                 level_start=None,
                 trend_start=None,
                 season_start=None,
                 delta=None,#SESM
                 adaptive_method=None,#SESM
                 ignore_zero=None,
                 expost_flag=None,
                 method=None
                ):

        super(_ExponentialSmoothingBase, self).__init__()

        self.model_selection = self._arg('model_selection', model_selection, (int, bool))
        self.forecast_model_name = self._arg('forecast_model_name', forecast_model_name, str)
        self.optimizer_time_budget = self._arg('optimizer_time_budget', optimizer_time_budget, int)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.optimizer_random_seed = self._arg('optimizer_random_seed', optimizer_random_seed, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.alpha = self._arg('alpha', alpha, float)
        self.beta = self._arg('beta', beta, float)
        self.gamma = self._arg('gamma', gamma, float)
        self.phi = self._arg('phi', phi, float)
        self.forecast_num = self._arg('forecast_num', forecast_num, int)
        self.seasonal_period = self._arg('seasonal_period', seasonal_period, int)
        self.seasonal = self._arg('seasonal', seasonal, (int, str))
        if isinstance(self.seasonal, str):
            self.seasonal = self._arg('seasonal', seasonal,
                                      dict(multiplicative=0, additive=1))
        self.initial_method = self._arg('initial_method', initial_method, int)
        self.training_ratio = self._arg('training_ratio', training_ratio, float)
        self.damped = self._arg('damped', damped, (int, bool))
        self.seasonality_criterion = self._arg('seasonality_criterion', seasonality_criterion, float)
        self.trend_test_method = self._arg('trend_test_method', trend_test_method, (int, str))
        if isinstance(self.trend_test_method, str):
            self.trend_test_method = self._arg('trend_test_method',
                                               trend_test_method,
                                               self.trend_test_map)
        self.trend_test_alpha = self._arg('trend_test_alpha', trend_test_alpha, float)
        self.alpha_min = self._arg('alpha_min', alpha_min, float)
        self.beta_min = self._arg('beta_min', beta_min, float)
        self.gamma_min = self._arg('gamma_min', gamma_min, float)
        self.phi_min = self._arg('phi_min', phi_min, float)
        self.alpha_max = self._arg('alpha_max', alpha_max, float)
        self.beta_max = self._arg('beta_max', beta_max, float)
        self.gamma_max = self._arg('gamma_max', gamma_max, float)
        self.phi_max = self._arg('phi_max', phi_max, float)
        self.prediction_confidence_1 = self._arg('prediction_confidence_1', prediction_confidence_1, float)
        self.prediction_confidence_2 = self._arg('prediction_confidence_2', prediction_confidence_2, float)
        self.level_start = self._arg('level_start', level_start, float)
        self.trend_start = self._arg('trend_start', trend_start, float)
        self.delta = self._arg('delta', delta, float)
        self.adaptive_method = self._arg('adaptive_method', adaptive_method, bool)
        self.ignore_zero = self._arg('ignore_zero', ignore_zero, bool)
        self.expost_flag = self._arg('expost_flag', expost_flag, bool)
        self.method = self._arg('method', method, int)

        # accuracy_measure for single/double/triple exp smooth
        accuracy_measure_list = {"mpe":"mpe", "mse":"mse", "rmse":"rmse", "et":"et",
                                 "mad":"mad", "mase":"mase", "wmape":"wmape",
                                 "smape":"smape", "mape":"mape"}
        if accuracy_measure is not None:
            if isinstance(accuracy_measure, str):
                accuracy_measure = [accuracy_measure]
            for acc in accuracy_measure:
                self._arg('accuracy_measure', acc.lower(), accuracy_measure_list)
            self.accuracy_measure = [acc.upper() for acc in accuracy_measure]
        else:
            self.accuracy_measure = None

        #check self.season_start which is a list of tuple. Each tuple has two elements and 1st element is int and 2nd is float
        self.season_start = self._arg('season_start', season_start, ListOfTuples)
        if self.season_start is not None:
            for element in self.season_start:
                if len(element) != 2:
                    msg = ('The length of each tuple of season_start should be 2!')
                    logger.error(msg)
                    raise ValueError(msg)

        if self.season_start is not None:
            for element in self.season_start:
                if not isinstance(element[0], int):
                    msg = ('The type of the first element of the tuple of season_start should be int!')
                    logger.error(msg)
                    raise ValueError(msg)
                if not isinstance(element[1], (float, int)):
                    msg = ('The type of the second element of the tuple of season_start should be float!')
                    logger.error(msg)
                    raise ValueError(msg)
        self.is_index_int = None
        self.forecast_start = None
        self.timedelta = None

    def _fit_predict(self, exp_smooth_function, data, key, endog):
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        if len(cols) < 2:
            msg = ("Input data should contain at least 2 columns: " +
                   "one for ID, another for raw data.")
            logger.error(msg)
            raise ValueError(msg)

        index = data.index
        key = self._arg('key', key, str)
        if key is not None and key not in cols:
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)

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
        if not self.is_index_int:
            data_ = _convert_index_from_timestamp_to_int(data_, key)
        try:
            self.forecast_start, self.timedelta = _get_forecast_starttime_and_timedelta(data, key, self.is_index_int)
        except Exception as err:
            logger.warning(err)
            pass

        function_map = {1:'PAL_SINGLE_EXPSMOOTH',
                        2:'PAL_DOUBLE_EXPSMOOTH',
                        3:'PAL_TRIPLE_EXPSMOOTH',
                        4:'PAL_AUTO_EXPSMOOTH',
                        5:'PAL_BROWN_EXPSMOOTH',
                        6:'PAL_CROSTON'}

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['FORECAST', 'STATISTICS']
        outputs = ['#PAL_EXP_SMOOTHING_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        forecast_tbl, stats_tbl = outputs
        param_rows = [
            ('MODELSELECTION', self.model_selection, None, None),
            ('FORECAST_MODEL_NAME', None, None, self.forecast_model_name),
            ('OPTIMIZER_TIME_BUDGET', self.optimizer_time_budget, None, None),
            ('MAX_ITERATION', self.max_iter, None, None),
            ('OPTIMIZER_RANDOM_SEED', self.optimizer_random_seed, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('ALPHA', None, self.alpha, None),
            ('BETA', None, self.beta, None),
            ('GAMMA', None, self.gamma, None),
            ('PHI', None, self.phi, None),
            ('FORECAST_NUM', self.forecast_num, None, None),
            ('CYCLE', self.seasonal_period, None, None),
            ('SEASONAL', self.seasonal, None, None),
            ('INITIAL_METHOD', self.initial_method, None, None),
            ('TRAINING_RATIO', None, self.training_ratio, None),
            ('DAMPED', self.damped, None, None),
            ('SEASONALITY_CRITERION', None, self.seasonality_criterion, None),
            ('TREND_TEST_METHOD', self.trend_test_method, None, None),
            ('TREND_TEST_ALPHA', None, self.trend_test_alpha, None),
            ('ALPHA_MIN', None, self.alpha_min, None),
            ('BETA_MIN', None, self.beta_min, None),
            ('GAMMA_MIN', None, self.gamma_min, None),
            ('PHI_MIN', None, self.phi_min, None),
            ('ALPHA_MAX', None, self.alpha_max, None),
            ('BETA_MAX', None, self.beta_max, None),
            ('GAMMA_MAX', None, self.gamma_max, None),
            ('PHI_MAX', None, self.phi_max, None),
            ('PREDICTION_CONFIDENCE_1', None, self.prediction_confidence_1, None),
            ('PREDICTION_CONFIDENCE_2', None, self.prediction_confidence_2, None),
            ('LEVEL_START', None, self.level_start, None),
            ('TREND_START', None, self.trend_start, None),
            ('DELTA', None, self.delta, None),#SESM
            ('ADAPTIVE_METHOD', self.adaptive_method, None, None),#SESM
            ('IGNORE_ZERO', self.ignore_zero, None, None),
            ('EXPOST_FLAG', self.expost_flag, None, None),
            ('METHOD', self.method, None, None)
        ]
        if self.accuracy_measure is not None:
            if isinstance(self.accuracy_measure, str):
                self.accuracy_measure = [self.accuracy_measure]
            for acc_measure in self.accuracy_measure:
                param_rows.extend([('ACCURACY_MEASURE', None, None, acc_measure)])
                param_rows.extend([('MEASURE_NAME', None, None, acc_measure)])
        if self.season_start is not None:
            param_rows.extend([('SEASON_START', element[0], element[1], None)
                               for element in self.season_start])

        pal_function = function_map[exp_smooth_function]

        if exp_smooth_function == 5:
            if check_pal_function_exist(conn, 'BROWN%INTERVAL%', like=True):
                pal_function = 'PAL_BROWN_EXPSMOOTH_INTERVAL'

        try:
            self._call_pal_auto(conn,
                                pal_function,
                                data_,
                                ParameterTable().with_data(param_rows),
                                forecast_tbl,
                                stats_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, forecast_tbl)
            try_drop(conn, stats_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, forecast_tbl)
            try_drop(conn, stats_tbl)
            raise
        self.stats_ = conn.table(stats_tbl)
        self.forecast_ = conn.table(forecast_tbl)
        if not self.is_index_int:
            if exp_smooth_function < 5:
                fct_ = conn.sql("""
                                SELECT ADD_SECONDS('{0}', ({1}-{9}) * {2}) AS {1},
                                {4},
                                {5},
                                {6},
                                {7},
                                {8}
                                FROM ({3})
                                """.format(self.forecast_start,
                                           quotename(self.forecast_.columns[0]),
                                           self.timedelta,
                                           self.forecast_.select_statement,
                                           quotename(self.forecast_.columns[1]),
                                           quotename(self.forecast_.columns[2]),
                                           quotename(self.forecast_.columns[3]),
                                           quotename(self.forecast_.columns[4]),
                                           quotename(self.forecast_.columns[5]),
                                           data.count() + 1))
            if exp_smooth_function == 5:
                if pal_function == 'PAL_BROWN_EXPSMOOTH_INTERVAL':
                    fct_ = conn.sql("""
                                    SELECT ADD_SECONDS('{0}', ({1}-{9}) * {2}) AS {1},
                                    {4},
                                    {5},
                                    {6},
                                    {7},
                                    {8}
                                    FROM ({3})
                                    """.format(self.forecast_start,
                                               quotename(self.forecast_.columns[0]),
                                               self.timedelta,
                                               self.forecast_.select_statement,
                                               quotename(self.forecast_.columns[1]),
                                               quotename(self.forecast_.columns[2]),
                                               quotename(self.forecast_.columns[3]),
                                               quotename(self.forecast_.columns[4]),
                                               quotename(self.forecast_.columns[5]),
                                               data.count() + 1))
                else:
                    fct_ = conn.sql("""
                                    SELECT ADD_SECONDS('{0}', ({1}-{5}) * {2}) AS {1},
                                    {4} FROM ({3})
                                    """.format(self.forecast_start,
                                               quotename(self.forecast_.columns[0]),
                                               self.timedelta,
                                               self.forecast_.select_statement,
                                               quotename(self.forecast_.columns[1]),
                                               data.count() + 1))
            if exp_smooth_function == 6:
                fct_ = conn.sql("""
                                SELECT ADD_SECONDS('{0}', ({1}-{6}) * {2}) AS {5},
                                {4} FROM ({3})
                                """.format(self.forecast_start,
                                           quotename(self.forecast_.columns[0]),
                                           self.timedelta,
                                           self.forecast_.select_statement,
                                           quotename(self.forecast_.columns[1]),
                                           quotename(key),
                                           data.count() + 1))
            self.forecast_ = fct_
        return self.forecast_

class SingleExponentialSmoothing(_ExponentialSmoothingBase):
    r"""
    Single Exponential Smoothing model is suitable to model the time series without trend and seasonality.
    In the model, the smoothed value is the weighted sum of previous smoothed value and previous observed value.
    PAL provides two simple exponential smoothing algorithms: single exponential smoothing and adaptive-response-rate simple exponential smoothing.
    The adaptive-response-rate single exponential smoothing algorithm may have an advantage over single exponential smoothing in that it allows the value of alpha to be modified.

    Parameters
    ----------

    alpha : float, optional
        The smoothing constant alpha for single exponential smoothing,
        or the initialization value for adaptive-response-rate single exponential smoothing.

        Valid range is (0, 1).

        Defaults to 0.1 for single exponential smoothing, and 0.2 for adaptive-response-rate single exponential smoothing.

    delta : float, optional
        Value of weighted for At and Mt(relative for the computation of adaptive smoothing parameter).

        The definitions of At and Mt are stated in PAL documentation for single exponential smoothing [1]_.

        Only valid when ``adaptive_method`` is True.

        Defaults to 0.2.

    forecast_num : int, optional
        Number of values to be forecast.

        Defaults to 0.

    adaptive_method : bool, optional

        - False: Single exponential smoothing.
        - True: Adaptive-response-rate single exponential smoothing.

        Defaults to False.

    accuracy_measure : str or list of str, optional

        The criterion used for the optimization.
        Options: "mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape".

        No default value.

    ignore_zero : bool, optional

        - False: Uses zero values in the input dataset when calculating "mpe" or "mape".
        - True: Ignores zero values in the input dataset when calculating "mpe" or "mape".

        Only valid when ``accuracy_measure`` is "mpe" or "mape".

        Defaults to False.

    expost_flag : bool, optional

        - False: Does not output the expost forecast, and just outputs the forecast values.
        - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    prediction_confidence_1 : float, optional
        Prediction confidence for interval 1.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.8.

    prediction_confidence_2 : float, optional
        Prediction confidence for interval 2.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.95.

    Attributes
    ----------
    forecast_ : DataFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    Returns
    -------
    DataFrame: Forecast values.

    Examples
    --------
    Input Dataframe df for SingleExponentialSmoothing:

    >>> df.collect()
        ID     RAW_DATA
    0    1        200.0
    1    2        135.0
    2    3        195.0
    3    4        197.5
    4    5        310.0
    5    6        175.0
    6    7        155.0
    7    8        130.0
    8    9        220.0
    9   10        277.5
    10  11        235.0

    Create a SingleExponentialSmoothing instance:

    >>> sesm = SingleExponentialSmoothing(adaptive_method=False,
                                          accuracy_measure='mse',
                                          alpha=0.1,
                                          delta=0.2,
                                          forecast_num=12,
                                          expost_flag=True,
                                          prediction_confidence_1=0.8,
                                          prediction_confidence_2=0.95)

    Perform fit_predict on the given data:

    >>> sesm.fit_predict(data=df)

    Output:

    >>> sesm.forecast_.collect().set_index('TIMESTAMP').head(3)
         TIMESTAMP       VALUE       PI1_LOWER     PI1_UPPER    PI2_LOWER    PI2_UPPER
    0           2          200             NaN           NaN          NaN          NaN
    1           3        193.5             NaN           NaN          NaN          NaN
    2           4        193.65            NaN           NaN          NaN          NaN

    >>> sesm.stats_.collect()
       STAT_NAME     STAT_VALUE
    0        MSE      3438.3321

    References
    ----------
    .. [1] PAL documentation for single exponential smoothing:
       https://help.sap.com/viewer/2cfbc5cf2bc14f028cfbe2a2bba60a50/2.0.06/en-US/ba4bb85a74d84c2b994aa7192cac3b1b.html
    """
    def __init__(self,
                 alpha=None,
                 delta=None,
                 forecast_num=None,
                 adaptive_method=None,
                 accuracy_measure=None,
                 ignore_zero=None,
                 expost_flag=None,
                 prediction_confidence_1=None,
                 prediction_confidence_2=None
                ):

        if delta is not None and adaptive_method is False:
            msg = ('delta is only valid when adaptive_method is True!')
            logger.error(msg)
            raise ValueError(msg)

        super(SingleExponentialSmoothing, self).__init__(
            alpha=alpha,
            delta=delta,
            forecast_num=forecast_num,
            adaptive_method=adaptive_method,
            accuracy_measure=accuracy_measure,
            ignore_zero=ignore_zero,
            expost_flag=expost_flag,
            prediction_confidence_1=prediction_confidence_1,
            prediction_confidence_2=prediction_confidence_2
            )

    def fit_predict(self, data, key=None, endog=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------
        data : DataFrame
            Input data. At least two columns, one is ID column, the other is raw data.

        key : str, optional
            The ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        """
        return super(SingleExponentialSmoothing, self)._fit_predict(exp_smooth_function=1,
                                                                    data=data,
                                                                    key=key,
                                                                    endog=endog)

class DoubleExponentialSmoothing(_ExponentialSmoothingBase):
    r"""
    Double Exponential Smoothing model is suitable to model the time series with trend but without seasonality.
    In the model there are two kinds of smoothed quantities: smoothed signal and smoothed trend.

    Parameters
    ----------

    alpha : float, optional
        Weight for smoothing. Value range: 0 < alpha < 1.

        Defaults to 0.1.

    beta : float, optional
        Weight for the trend component. Value range: 0 < beta < 1.

        Defaults to 0.1.

    forecast_num : int, optional
        Number of values to be forecast.

        Defaults to 0.

    phi : float, optional
        Value of the damped smoothing constant phi (0 < phi < 1).

        Defaults to 0.1.

    damped : bool, optional
        Specifies whether or not to use damped trend method.

        - False: No, uses the Holt's linear trend method.
        - True: Yes, use damped trend method.

        Defaults to False.

    accuracy_measure : str or list of str, optional

        The criterion used for the optimization.
        Options: "mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape".

        No default value.

    ignore_zero : bool, optional

        - False: Uses zero values in the input dataset when calculating "mpe" or "mape".
        - True: Ignores zero values in the input dataset when calculating "mpe" or "mape".

        Only valid when ``accuracy_measure`` is "mpe" or "mape".

        Defaults to False.

    expost_flag : bool, optional

        - False: Does not output the expost forecast, and just outputs the forecast values.
        - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    prediction_confidence_1 : float, optional
        Prediction confidence for interval 1.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.8.

    prediction_confidence_2 : float, optional
        Prediction confidence for interval 2.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.95.

    Attributes
    ----------
    forecast_ : DataFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    Returns
    -------
    DataFrame: Forecast values.

    Examples
    --------
    Input Dataframe df for DoubleExponentialSmoothing:

    >>> df.collect()
            ID    RAW_DATA
    0        1       143.0
    1        2       152.0
    2        3       161.0
    3        4       139.0
    ...
    20      21       223.0
    21      22       242.0
    22      23       239.0
    23      24       266.0

    Create a DoubleExponentialSmoothing instance:

    >>> desm = DoubleExponentialSmoothing(alpha=0.501,
                                          beta=0.072,
                                          forecast_num=6,
                                          phi=None,
                                          damped=None,
                                          accuracy_measure='mse',
                                          ignore_zero=None,
                                          expost_flag=None,
                                          prediction_confidence_1=0.8,
                                          prediction_confidence_2=0.95)

    Perform fit_predict on the given data:

    >>> desm.fit_predict(data=df)

    Output:

    >>> desm.forecast_.collect().set_index('TIMESTAMP').head(3)
        TIMESTAMP        VALUE   PI1_LOWER     PI1_UPPER    PI2_LOWER    PI2_UPPER
    0           2          152         NaN           NaN          NaN          NaN
    1           3          161         NaN           NaN          NaN          NaN
    2           4          170         NaN           NaN          NaN          NaN

    >>> desm.stats_.collect()
       STAT_NAME       STAT_VALUE
    0        MSE      274.8960228
    """
    def __init__(self,
                 alpha=None,
                 beta=None,
                 forecast_num=None,
                 phi=None,
                 damped=None,
                 accuracy_measure=None,
                 ignore_zero=None,
                 expost_flag=None,
                 prediction_confidence_1=None,
                 prediction_confidence_2=None
                ):

        super(DoubleExponentialSmoothing, self).__init__(
            alpha=alpha,
            beta=beta,
            forecast_num=forecast_num,
            phi=phi,
            damped=damped,
            accuracy_measure=accuracy_measure,
            ignore_zero=ignore_zero,
            expost_flag=expost_flag,
            prediction_confidence_1=prediction_confidence_1,
            prediction_confidence_2=prediction_confidence_2
            )

    def fit_predict(self, data, key=None, endog=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------
        data : DataFrame
            Input data. At least two columns, one is ID column, the other is raw data.

        key : str, optional
            The ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        """
        return super(DoubleExponentialSmoothing, self)._fit_predict(exp_smooth_function=2,
                                                                    data=data,
                                                                    key=key,
                                                                    endog=endog)

class TripleExponentialSmoothing(_ExponentialSmoothingBase):
    r"""
    Triple exponential smoothing is used to handle the time series data containing a seasonal component.

    Parameters
    ----------

    alpha : float, optional
        Weight for smoothing. Value range: 0 < alpha < 1.

        Defaults to 0.1.

    beta : float, optional
        Weight for the trend component. Value range: 0 <= beta < 1.

        Defaults to 0.1.

    gamma : float, optional
        Weight for the seasonal component. Value range: 0 < gamma < 1.

        Defaults to 0.1.

    seasonal_period : int, optional
        Length of a seasonal_period(should be greater than 1).

        For example, the ``seasonal_period`` of quarterly data is 4,
        and the ``seasonal_period`` of monthly data is 12.

        Defaults to 2.

    forecast_num : int, optional
        Number of values to be forecast.

        Defaults to 0.

    seasonal : {'multiplicative', 'additive'}, optional
        Specifies the type of model for triple exponential smoothing.

            - 'multiplicative': Multiplicative triple exponential smoothing.
            - 'additive': Additive triple exponential smoothing.

        When ``seasonal`` is set to 'addtive', the default value of initial_method is 1;
        When ``seasonal`` is set to 'multiplicative', the default value of initial_method is 0.

        Defaults to 'multiplicative'.

    initial_method : int, optional
        Initialization method for the trend and seasonal components.

        Defaults to 0 or 1, depending the setting of ``seasonal``.

    phi : float, optional
        Value of the damped smoothing constant phi (0 < phi < 1).

        Defaults to 0.1.

    damped : bool, optional
        Specifies whether or not to use damped trend method.

        - False: No, uses the Holt's linear trend method.
        - True: Yes, use damped trend method.

        Defaults to False.

    accuracy_measure : str or list of str, optional

        The criterion used for the optimization.
        Options: "mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape".

        No default value.

    ignore_zero : bool, optional

        - False: Uses zero values in the input dataset when calculating "mpe" or "mape".
        - True: Ignores zero values in the input dataset when calculating "mpe" or "mape".

        Only valid when ``accuracy_measure`` is "mpe" or "mape".

        Defaults to False.

    expost_flag : bool, optional

        - False: Does not output the expost forecast, and just outputs the forecast values.
        - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    level_start : float, optional
        The initial value for level component S.

        If this value is not provided, it will be calculated in the way as described in Triple Exponential Smoothing.

        ``level_start`` cannot be zero. If it is set to zero, 0.0000000001 will be used instead.

    trend_start : float, optional
        The initial value for trend component B.

    season_start: list of tuple, optional
        A list of initial values for seasonal component C.

        Two values must be provided for each cycle:

         - Cycle ID: An int which represents which cycle the initial value is used for.
         - Initial value: A double precision number which represents the initial value for the corresponding cycle.

        For example: To give the initial value 0.5 to the 3rd cycle, insert list of tuple [(3,5)] into the parameter table.

    prediction_confidence_1 : float, optional
        Prediction confidence for interval 1.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.8.

    prediction_confidence_2 : float, optional
        Prediction confidence for interval 2.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.95.

    Attributes
    ----------
    forecast_ : DataFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    Returns
    -------
    DataFrame: Forecast values.

    Examples
    --------
    Input Dataframe df for TripleExponentialSmoothing:

    >>> df.collect()
        ID    RAW_DATA
    0    1       362.0
    1    2       385.0
    2    3       432.0
    3    4       341.0
    4    5       382.0
    5    6       409.0
    6    7       498.0
    7    8       387.0
    8    9       473.0
    9   10       513.0
    10  11       582.0
    11  12       474.0
    12  13       544.0
    13  14       582.0
    14  15       681.0
    15  16       557.0
    16  17       628.0
    17  18       707.0
    18  19       773.0
    19  20       592.0
    20  21       627.0
    21  22       725.0
    22  23       854.0
    23  24       661.0

    Create a TripleExponentialSmoothing instance:

    >>> tesm = TripleExponentialSmoothing(alpha=0.822,
                                          beta=0.055,
                                          gamma=0.055,
                                          seasonal_period=4,
                                          forecast_num=6,
                                          seasonal=0,
                                          initial_method=0,
                                          phi=None,
                                          damped=None,
                                          accuracy_measure='mse',
                                          ignore_zero=None,
                                          expost_flag=True,
                                          level_start=None,
                                          trend_start=None,
                                          season_start=None,
                                          prediction_confidence_1=0.8,
                                          prediction_confidence_2=0.95)

    Perform fit_predict on the given data:

    >>> tesm.fit_predict(data=df)

    Output:

    >>> tesm.forecast_.collect().set_index('TIMESTAMP').head(3)
      TIMESTAMP           VALUE   PI1_LOWER    PI1_UPPER   PI2_LOWER    PI2_UPPER
    0        5       371.288158         NaN          NaN         NaN          NaN
    1        6       414.636207         NaN          NaN         NaN          NaN
    2        7       471.431808         NaN          NaN         NaN          NaN

    >>> tesm.stats_.collect()
       STAT_NAME        STAT_VALUE
    0        MSE        616.541542
    """

    def __init__(self,
                 alpha=None,
                 beta=None,
                 gamma=None,
                 seasonal_period=None,
                 forecast_num=None,
                 seasonal=None,
                 initial_method=None,
                 phi=None,
                 damped=None,
                 accuracy_measure=None,
                 ignore_zero=None,
                 expost_flag=None,
                 level_start=None,
                 trend_start=None,
                 season_start=None,
                 prediction_confidence_1=None,
                 prediction_confidence_2=None
                ):

        super(TripleExponentialSmoothing, self).__init__(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            seasonal_period=seasonal_period,
            forecast_num=forecast_num,
            seasonal=seasonal,
            initial_method=initial_method,
            phi=phi,
            damped=damped,
            accuracy_measure=accuracy_measure,
            ignore_zero=ignore_zero,
            expost_flag=expost_flag,
            level_start=level_start,
            trend_start=trend_start,
            season_start=season_start,
            prediction_confidence_1=prediction_confidence_1,
            prediction_confidence_2=prediction_confidence_2
            )

    def fit_predict(self, data, key=None, endog=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------
        data : DataFrame
            Input data. At least two columns, one is ID column, the other is raw data.

        key : str, optional
            The ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        """
        return super(TripleExponentialSmoothing, self)._fit_predict(exp_smooth_function=3,
                                                                    data=data,
                                                                    key=key,
                                                                    endog=endog)

class AutoExponentialSmoothing(_ExponentialSmoothingBase):
    r"""
    Auto exponential smoothing (previously named forecast smoothing) is used to calculate optimal parameters of a set of smoothing functions in SAP HANA PAL,
    including Single Exponential Smoothing, Double Exponential Smoothing, and Triple Exponential Smoothing.

    Parameters
    ----------

    model_selection : bool, optional
        Specifies whether the algorithms will perform model selection or not.

            - True: the algorithm will select the best model among Single/Double/Triple/
              Damped Double/Damped Triple Exponential Smoothing models.
            - False: the algorithm will not perform the model selection.

        If ``forecast_model_name`` is set, the model defined by forecast_model_name will be used.

        Defaults to False.

    forecast_model_name : str, optional
        Name of the statistical model used for calculating the forecast.

        - 'SESM': Single Exponential Smoothing.
        - 'DESM': Double Exponential Smoothing.
        - 'TESM': Triple Exponential Smoothing.

        This parameter must be set unless ``model_selection`` is set to 1.

    optimizer_time_budget : int, optional
        Time budget for Nelder-Mead optimization process.

        The time unit is second and the value should be larger than zero.

        Defaults to 1.

    max_iter : int, optional
        Maximum number of iterations for simulated annealing.

        Defaults to 100.

    optimizer_random_seed : int, optional
        Random seed for simulated annealing.

        The value should be larger than zero.

        Defaults to system time.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The ratio of available threads.

          - 0: single thread.
          - 0~1: percentage.
          - Others: heuristically determined.

        Defaults to 1.0.

    alpha : float, optional
        Weight for smoothing. Value range: 0 < alpha < 1.

        Default value is computed automatically.

    beta : float, optional
        Weight for the trend component. Value range: 0 <= beta < 1.

        If it is not set, the optimized value will be computed automatically.

        Only valid when the model is set by user or identified by the algorithm as 'DESM' or 'TESM'.

        Value 0 is allowed under TESM model only.

        Defaults value is computed automatically.

    gamma : float, optional
        Weight for the seasonal component. Value range: 0 < gamma < 1.
        Only valid when the model is set by user or identified by the algorithm as TESM.

        Default value is computed automatically.

    phi : float, optional
        Value of the damped smoothing constant phi (0 < phi < 1).
        Only valid when the model is set by user or identified by the algorithm as a damped model.

        Default value is computed automatically.

    forecast_num : int, optional
        Number of values to be forecast.
        Defaults to 0.

    seasonal_period : int, optional
        Length of a seasonal_period (L > 1).

        For example, the ``seasonal_period`` of quarterly data is 4,
        and the ``seasonal_period`` of monthly data is 12.

        Only valid when the model is set by user or identified by the algorithm as 'TESM'.

        Default value is computed automatically.

    seasonal : {'multiplicative', 'additive'}, optional
        Specifies the type of model for triple exponential smoothing.

            - 'multiplicative': Multiplicative triple exponential smoothing.
            - 'additive': Additive triple exponential smoothing.

        When ``seasonal`` is set to 'addtive', the default value of initial_method is 1;
        When ``seasonal`` is set to 'multiplicative', the default value of initial_method is 0.

        Defaults to 'multiplicative'.

    initial_method : int, optional
        Initialization method for the trend and seasonal components.

        Refer to :class:`~hana_ml.algorithms.pal.tsa.exponential_smoothing.TripleExponentialSmoothing` for detailed information on initialization method.

        Only valid when the model is set by user or identified by the algorithm as 'TESM'.

        Defaults to 0 or 1.

    training_ratio : float, optional
        The ratio of training data to the whole time series.

        Assuming the size of time series is N, and the training ratio is r,
        the first N*r time series is used to train, whereas only the latter N*(1-r) one
        is used to test.

        If this parameter is set to 0.0 or 1.0, or the resulting training data
        (N*r) is less than 1 or equal to the size of time series, no train-and-test procedure is
        carried out.

        Defaults to 1.0.

    damped : int, optional
        For DESM:

          - False: Uses the Holt's linear method.
          - True: Uses the additive damped trend Holt's linear method.

        For TESM:

          - False: Uses the Holt Winter method.
          - True: Uses the additive damped seasonal Holt Winter method.

        If ``model_selection`` is set to 1, the default value will be computed automatically.
        Otherwise, the default value is False.

    accuracy_measure : str, {'mse', 'mape'}, optional
        The criterion used for the optimization.

        Defaults to 'mse'.

    seasonality_criterion : float, optional
        The criterion of the auto-correlation coefficient for accepting seasonality,
        in the range of (0, 1).

        The larger it is, the less probable a time series is
        regarded to be seasonal.

        Only valid when ``forecast_model_name`` is 'TESM' or model_selection
        is set to 1, and ``seasonal_period`` is not defined.

        Defaults to 0.5.

    trend_test_method : {'mk', 'difference-sign'}, optional

        - 'mk': Mann-Kendall test.
        - 'difference-sign': Difference-sign test.

        Defaults to 'mk'.

    trend_test_alpha : float, optional
        Tolerance probability for trend test. The value range is (0, 0.5).

        Only valid when ``model_selection`` is set to 1.

        Defaults to 0.05.

    alpha_min : float, optional
        Sets the minimum value of alpha.

        Only valid when ``alpha`` is not defined.

        Defaults to 0.0000000001.

    beta_min : float, optional
        Sets the minimum value of beta.

        Only valid when ``beta`` is not defined.

        Defaults to 0.0000000001.

    gamma_min : float, optional
        Sets the minimum value of gamma.

        Only valid when ``gamma`` is not defined.

        Defaults to 0.0000000001.

    phi_min : float, optional
        Sets the minimum value of phi.

        Only valid when ``phi`` is not defined.

        Defaults to 0.0000000001.

    alpha_max : float, optional
        Sets the maximum value of alpha.

        Only valid when ``alpha`` is not defined.

        Defaults to 1.0.

    beta_max : float, optional
        Sets the maximum value of beta.

        Only valid when ``beta`` is not defined.

        Defaults to 1.0.

    gamma_max : float, optional
        Sets the maximum value of gamma.

        Only valid when ``gamma`` is not defined.

        Defaults to 1.0.

    phi_max : float, optional
        Sets the maximum value of phi.

        Only valid when ``phi`` is not defined.

        Defaults to 1.0.

    prediction_confidence_1 : float, optional
        Prediction confidence for interval 1.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.8.

    prediction_confidence_2 : float, optional
        Prediction confidence for interval 2.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to is 0.95.

    level_start : float, optional
        The initial value for level component S.

        If this value is not provided, it will be calculated in the way as described in :class:`~hana_ml.algorithms.pal.tsa.exponential_smoothing.TripleExponentialSmoothing`.

        Notice that ``level_start`` cannot be zero.

        If it is set to zero, 0.0000000001 will be used instead.

    trend_start : float, optional
        The initial value for trend component B.

        If this value is not provided, it will be calculated in the way as described in :class:`~hana_ml.algorithms.pal.tsa.exponential_smoothing.TripleExponentialSmoothing`.

    season_start : list of tuple, optional
        A list of initial values for seasonal component C.

        Two values must be provided for each cycle:

         - Cycle ID: An int which represents which cycle the initial value is used for.
         - Initial value: A double precision number which represents the initial value for the corresponding cycle.

        For example: To give the initial value 0.5 to the 3rd cycle, insert list of tuple [(3,5)] into the parameter table.


    Attributes
    ----------
    forecast_ : DataFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    Returns
    -------
    DataFrame: Forecast values.

    Examples
    --------
    Input Dataframe df for AutoExponentialSmoothing:

    >>> df.collect()
        TIMESTAMP       Y
    0           1     362
    1           2     385
    2           3     432
    3           4     341
    4           5     382
    ...
    20         21     627
    21         22     725
    22         23     854
    23         24     661

    Create AutoExponentialSmoothing instance:

    >>> autoExp = time_series.AutoExponentialSmoothing(forecast_model_name='TESM',
                                                       alpha=0.4,
                                                       beta=0.4,
                                                       gamma=0.4,
                                                       seasonal_period=4,
                                                       forecast_num=3,
                                                       seasonal='multiplicative',
                                                       initial_method=1,
                                                       training_ratio=0.75)

    Perform fit on the given data:

    >>> autoExp.fit(data=df)

    Output:

    >>> autoExp.forecast_.collect().set_index('TIMESTAMP').head(6)
       TIMESTAMP        VALUE   PI1_LOWER    PI1_UPPER   PI2_LOWER    PI2_UPPER
    0          1   320.018502         NaN          NaN         NaN          NaN
    1          2   374.225113         NaN          NaN         NaN          NaN
    2          3   458.649782         NaN          NaN         NaN          NaN
    3          4   364.376078         NaN          NaN         NaN          NaN
    4          5   416.009008         NaN          NaN         NaN          NaN


    >>> autoExp.stats_.collect().head(4)
                      STAT_NAME         STAT_VALUE
    0                       MSE   467.811415778471
    1      NUMBER_OF_ITERATIONS                110
    2   SA_NUMBER_OF_ITERATIONS                100
    3   NM_NUMBER_OF_ITERATIONS                 10

    """

    def __init__(self,
                 model_selection=None,# Auto ESM
                 forecast_model_name=None,# Auto ESM
                 optimizer_time_budget=None,# Auto ESM
                 max_iter=None,# Auto ESM
                 optimizer_random_seed=None,# Auto ESM
                 thread_ratio=None,# Auto ESM
                 alpha=None,
                 beta=None,
                 gamma=None,
                 phi=None,
                 forecast_num=None,
                 seasonal_period=None,
                 seasonal=None,
                 initial_method=None,
                 training_ratio=None,
                 damped=None,
                 accuracy_measure=None,
                 seasonality_criterion=None,# Auto ESM
                 trend_test_method=None,# Auto ESM
                 trend_test_alpha=None,# Auto ESM
                 alpha_min=None, # Auto ESM
                 beta_min=None,# Auto ESM
                 gamma_min=None,# Auto ESM
                 phi_min=None,# Auto ESM
                 alpha_max=None,# Auto ESM
                 beta_max=None,# Auto ESM
                 gamma_max=None,# Auto ESM
                 phi_max=None,# Auto ESM
                 prediction_confidence_1=None,
                 prediction_confidence_2=None,
                 level_start=None,
                 trend_start=None,
                 season_start=None
                ):
        if accuracy_measure is not None:
            if isinstance(accuracy_measure, str):
                accuracy_measure = [accuracy_measure]
            if len(accuracy_measure) != 1:
                msg = "Please select accuracy_measure from 'mse' OR 'mape'!"
                logger.error(msg)
                raise ValueError(msg)
            self._arg('accuracy_measure', accuracy_measure[0].lower(), {'mse':'mse', 'mape':'mape'})
            accuracy_measure = accuracy_measure[0].lower()

        super(AutoExponentialSmoothing, self).__init__(
            model_selection,
            forecast_model_name,
            optimizer_time_budget,
            max_iter,
            optimizer_random_seed,
            thread_ratio,
            alpha,
            beta,
            gamma,
            phi,
            forecast_num,
            seasonal_period,
            seasonal,
            initial_method,
            training_ratio,
            damped,
            accuracy_measure,
            seasonality_criterion,
            trend_test_method,
            trend_test_alpha,
            alpha_min,
            beta_min,
            gamma_min,
            phi_min,
            alpha_max,
            beta_max,
            gamma_max,
            phi_max,
            prediction_confidence_1,
            prediction_confidence_2,
            level_start,
            trend_start,
            season_start,
            None,
            None,
            None,
            None
            )

    def fit_predict(self, data, key=None, endog=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------
        data : DataFrame
            Input data. At least two columns, one is ID column, the other is raw data.

        key : str, optional
            The ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        """
        if self.training_ratio is None:
            self.training_ratio = 1.0
        rows = data.count() * self.training_ratio
        half_row = rows/2

        if self.seasonal_period is not None and self.seasonal_period > half_row:
            msg = ('seasonal_period should be smaller than' +
                   ' 1/2(row number * training_ratio) of data!')
            logger.error(msg)
            raise ValueError(msg)

        return super(AutoExponentialSmoothing, self)._fit_predict(exp_smooth_function=4, data=data,
                                                                  key=key, endog=endog)

class BrownExponentialSmoothing(_ExponentialSmoothingBase):
    r"""
    The brown exponential smoothing model is suitable to model the time series with trend but without seasonality.
    In PAL, both non-adaptive and adaptive brown linear exponential smoothing are provided.

    Parameters
    ----------

    alpha : float, optional
        The smoothing constant alpha for brown exponential smoothing or
        the initialization value for adaptive brown exponential smoothing (0 < alpha < 1).

          - Defaults to 0.1 when Brown exponential smoothing
          - Defaults to 0.2 when Adaptive brown exponential smoothing

    delta : float, optional
        Value of weighted for At and Mt.

        Only valid when ``adaptive_method`` is True.

        Defaults to 0.2

    forecast_num : int, optional
        Number of values to be forecast.

        Defaults to 0.

    adaptive_method : bool, optional

        - False: Brown exponential smoothing.
        - True: Adaptive brown exponential smoothing.

        Defaults to False.

    accuracy_measure : str or list of str, optional

        The criterion used for the optimization.
        Options: "mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape".

        No default value.

    ignore_zero : bool, optional

        - False: Uses zero values in the input dataset when calculating "mpe" or "mape".
        - True: Ignores zero values in the input dataset when calculating "mpe" or "mape".

        Only valid when ``accuracy_measure`` is "mpe" or "mape".

        Defaults to False.

    expost_flag : bool, optional
         - False: Does not output the expost forecast, and just outputs the forecast values.
         - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    prediction_confidence_1 : float, optional
        Prediction confidence for interval 1.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.8.

    prediction_confidence_2 : float, optional
        Prediction confidence for interval 2.

        Only valid when the upper and lower columns are provided in the result table.

        Defaults to 0.95.

    Attributes
    ----------
    forecast_ : DateFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    Returns
    -------
    DataFrame: Forecast values.

    Examples
    --------
    Input dataframe df for BrownExponentialSmoothing:

    >>> df.collect()
          ID        RAWDATA
    0      1          143.0
    1      2          152.0
    2      3          161.0
    3      4          139.0
    4      5          137.0
    ...
    20    21          223.0
    21    22          242.0
    22    23          239.0
    23    24          266.0

    Create BrownExponentialSmoothing instance:

    >>> brown_exp_smooth = BrownExponentialSmoothing(alpha=0.1,
                                                     delta=0.2,
                                                     forecast_num=6,
                                                     adaptive_method=False,
                                                     accuracy_measure='mse',
                                                     ignore_zero=0,
                                                     expost_flag=1)

    Perform fit on the given data:

    >>> brown_exp_smooth.fit_predict(data=df)

    Output:

    >>> brown_exp_smooth.forecast_.collect().set_index('TIMESTAMP').head(6)
      TIMESTAMP      VALUE
    0         2  143.00000
    1         3  144.80000
    2         4  148.13000
    3         5  146.55600
    4         6  144.80550
    5         7  150.70954


    >>> brown_exp_smooth.stats_.collect()
        STAT_NAME       STAT_VALUE
    0         MSE       474.142004

    """
    def __init__(self,
                 alpha=None,
                 delta=None,
                 forecast_num=None,
                 adaptive_method=None,
                 accuracy_measure=None,
                 ignore_zero=None,
                 expost_flag=None,
                 prediction_confidence_1=None,
                 prediction_confidence_2=None):

        if delta is not None and adaptive_method is False:
            msg = 'delta is only valid when adaptive_method is True!'
            logger.error(msg)
            raise ValueError(msg)

        super(BrownExponentialSmoothing, self).__init__(alpha=alpha,
                                                        delta=delta,
                                                        forecast_num=forecast_num,
                                                        adaptive_method=adaptive_method,
                                                        accuracy_measure=accuracy_measure,
                                                        ignore_zero=ignore_zero,
                                                        expost_flag=expost_flag,
                                                        prediction_confidence_1=prediction_confidence_1,
                                                        prediction_confidence_2=prediction_confidence_2)

    def fit_predict(self, data, key=None, endog=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------
        data : DataFrame
            Input data. At least two columns, one is ID column, the other is raw data.

        key : str, optional
            The ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        """
        return super(BrownExponentialSmoothing, self)._fit_predict(exp_smooth_function=5,
                                                                   data=data,
                                                                   key=key,
                                                                   endog=endog)

class Croston(_ExponentialSmoothingBase):
    r"""
    The Croston method is a forecast strategy for products with intermittent demand.
    The Croston method consists of two steps. First, separate exponential smoothing estimates are made of the average size of a demand.
    Second, the average interval between demands is calculated. This is then used in a form of the constant model to predict the future demand.

    Parameters
    ----------

    alpha : float, optional
        Value of the smoothing constant alpha (0 < alpha < 1).

        Defaults to 0.1.

    forecast_num : int, optional
        Number of values to be forecast.

        When it is set to 1, the algorithm only forecasts one value.

        Defaults to 0.

    method : str, optional

        - 'sporadic': Use the sporadic method.
        - 'constant': Use the constant method.

        Defaults to 'sporadic'.

    accuracy_measure : str or list of str, optional

        The criterion used for the optimization.
        Options: "mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape".

        No default value.

    ignore_zero : bool, optional

        - False: Uses zero values in the input dataset when calculating "mpe" or "mape".
        - True: Ignores zero values in the input dataset when calculating "mpe" or "mape".

        Only valid when ``accuracy_measure`` is "mpe" or "mape".

        Defaults to False.

    expost_flag : bool, optional

        - False: Does not output the expost forecast, and just outputs the forecast values.
        - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    Attributes
    ----------
    forecast_ : DateFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    Returns
    -------
    DataFrame: Forecast values.

    Examples
    --------
    Input dataframe df for Croston:

    >>> df.collect()
        ID       RAWDATA
    0    0           0.0
    1    1           1.0
    2    2           4.0
    3    3           0.0
    4    4           0.0
    5    5           0.0
    6    6           5.0
    7    7           3.0
    8    8           0.0
    9    9           0.0
    10  10           0.0

    Create Croston instance:

    >>> croston = Croston(alpha=0.1,
                          forecast_num=1,
                          method='sporadic',
                          accuracy_measure='mape')

    Perform fit on the given data:

    >>> croston.fit_predict(data=df)

    Output:

    >>> croston.forecast_.collect().set_index('ID').head(6)
           ID           RAWDATA
    0       0          0.000000
    1       1          3.025000
    2       2          3.122500
    3       3          0.000000
    4       4          0.000000
    5       5          0.000000


    >>> croston.stats_.collect()
        STAT_NAME                STAT_VALUE
    0        MAPE        0.2432181818181818

    """
    def __init__(self,
                 alpha=None,
                 forecast_num=None,
                 method=None,
                 accuracy_measure=None,
                 ignore_zero=None,
                 expost_flag=None):
        method = self._arg('method', method,
                           {'sporadic': 0, 'constant': 1})
        if alpha is None:
            alpha = 0.1
        super(Croston, self).__init__(alpha=alpha,
                                      forecast_num=forecast_num,
                                      accuracy_measure=accuracy_measure,
                                      ignore_zero=ignore_zero,
                                      expost_flag=expost_flag,
                                      method=method)

    def fit_predict(self, data, key=None, endog=None):
        """
        Fit and predict based on the given time series.

        Parameters
        ----------
        data : DataFrame
            Input data. At least two columns, one is ID column, the other is raw data.

        key : str, optional
            The ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first non-ID column.

        """
        return super(Croston, self)._fit_predict(exp_smooth_function=6, data=data, key=key,
                                                 endog=endog)
