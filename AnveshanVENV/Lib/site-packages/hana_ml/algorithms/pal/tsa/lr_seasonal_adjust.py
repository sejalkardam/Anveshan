"""
This module contains Python wrapper for PAL linear regression
with damped trend and seasonal adjust algorithm.

The following class is available:

    * :class:`LR_seasonal_adjust`
"""
#pylint: disable=too-many-lines, line-too-long, invalid-name, relative-beyond-top-level,  bare-except
#pylint: disable=too-many-instance-attributes, attribute-defined-outside-init, too-many-branches
#pylint: disable=too-many-locals, too-few-public-methods, too-many-arguments, too-many-statements
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
from hana_ml.algorithms.pal.tsa.arima import _convert_index_from_timestamp_to_int, _is_index_int
from hana_ml.algorithms.pal.tsa.arima import _get_forecast_starttime_and_timedelta
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    try_drop,
    require_pal_usable
)
logger = logging.getLogger(__name__)



class LR_seasonal_adjust(PALBase):
    """
    Linear regression with damped trend and seasonal adjust is an approach for forecasting when a time series presents a trend.

    Parameters
    ----------
    forecast_length : int, optional
        Specifies the length of forecast results.

        Defaults to 1.

    trend : float, optional
        Specifies the damped trend factor, with valid range (0, 1].

        Defaults to 1.

    affect_future_only : bool, optional
        Specifies whether the damped trend affect future only or it affects the entire history.

        Defaults to True.

    seasonality : int, optional
        Specifies whether the data represents seasonality.

          - 0: Non-seasonality.
          - 1: Seasonality exists and user inputs the value of periods.
          - 2: Automatically detects seasonality.

        Defaults to 0.

    seasonal_period : int, optional

        Length of seasonal_period.seasonal_period is only valid when seasonality is 1.

        If this parameter is not specified, the seasonality value will be changed from 1 to 2,
        that is, from user-defined to automatically-detected.

        No default value.

    seasonal_handle_method : {'average', 'lr'}, optional

        Method used for calculating the index value in the seasonal_period.

          - 'average': Average method.
          - 'lr': Fitting linear regression.

        Defaults to 'average'.

    accuracy_measure : str or list of str, optional

        The criterion used for the optimization.
        Options: "mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape".

        No default value.


    ignore_zero : bool, optional
         - False: Uses zero values in the input dataset when calculating 'mpe' or 'mape'.
         - True: Ignores zero values in the input dataset when calculating 'mpe' or 'mape'.

        Only valid when ``accuracy_measure`` is 'mpe' or 'mape'.

        Defaults to False.

    expost_flag : bool, optional

         - False: Does not output the expost forecast, and just outputs the forecast values.
         - True: Outputs the expost forecast and the forecast values.

       Defaults to True.

    Attributes
    ----------

    forecast_ : DataFrame

        Forecast values.

    stats_ : DataFrame

        Statistics analysis content.

    Examples
    --------

    Input Dataframe df:

    >>> df.collect()
             TIMESTAMP    Y
             1            5384
             2            8081
             3            10282
             4            9156
             5            6118
             6            9139
             7            12460
             8            10717
             9            7825
             10           9693
             11           15177
             12           10990

    Create a LR_seasonal_adjust instance:

    >>> lr = LR_seasonal_adjust(forecast_length=10,
                                trend=0.9, affect_future_only=True,
                                seasonality=1, seasonal_period=4,
                                accuracy_measure='mse')

    Perform fit_predict on the given data:

    >>> lr.fit_predict(data=df)

    Output:

    >>> lr.forecast_.collect().set_index('TIMESTAMP').head(3)
        TIMESTAMP    VALUE
        1            5328.171741
        2            7701.608247
        3            11248.606332

    >>> lr.stats_.collect()
           STAT_NAME        STAT_VALUE
           Intercept        7626.072428
           Slope            301.399114
           Periods          4.000000
           Index0           0.672115
           Index1           0.935925
           Index2           1.318669
           Index3           1.073290
           MSE              332202.479082
           HandleZero       0.000000
    """
    accuracy_measure_list = ["mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape"]
    seasonal_handle_method_map = {'average':0, 'lr':1}

    def __init__(self,
                 forecast_length=None,
                 trend=None,
                 affect_future_only=None,
                 seasonality=None,
                 seasonal_period=None,
                 accuracy_measure=None,
                 seasonal_handle_method=None,
                 expost_flag=None,
                 ignore_zero=None):
        super(LR_seasonal_adjust, self).__init__()
        self.forecast_length = self._arg('forecast_length', forecast_length, int)
        self.trend = self._arg('trend', trend, float)
        self.affect_future_only = self._arg('affect_future_only', affect_future_only, bool)
        self.seasonality = self._arg('seasonality', seasonality, int)
        self.seasonal_period = self._arg('seasonal_period', seasonal_period, int)
        self.seasonal_handle_method = self._arg('seasonal_handle_method', seasonal_handle_method, self.seasonal_handle_method_map)
        self.expost_flag = self._arg('expost_flag', expost_flag, bool)
        self.ignore_zero = self._arg('ignore_zero', ignore_zero, bool)

        # accuracy_measure for single/double/triple exp smooth
        accuracy_measure_list = ["mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape"]
        measure_list = ["mpe", "mape"]
        if accuracy_measure is not None:
            if isinstance(accuracy_measure, str):
                accuracy_measure = [accuracy_measure]
            for acc in accuracy_measure:
                acc = acc.lower()
                if acc not in accuracy_measure_list:
                    msg = ('Please select accuracy_measure from the list ["mpe", "mse", "rmse", "et", "mad", "mase", "wmape", "smape", "mape"]!')
                    logger.error(msg)
                    raise ValueError(msg)
                #for ignore_zero method check
                if ignore_zero is not None and acc not in measure_list:
                    msg = ('Please select accuracy_measure from "mpe" and "mape" when ignore_zero is not None!')
                    logger.error(msg)
                    raise ValueError(msg)
            self.accuracy_measure = [acc.upper() for acc in accuracy_measure]
        else:
            self.accuracy_measure = None

        measure_list = ["mpe", "mape"]
        if self.ignore_zero is not None and self.accuracy_measure not in measure_list:
            msg = ('Please select accuracy_measure from "mpe" and "mape" when ignore_zero is not None!')
            logger.error(msg)
            raise ValueError(msg)

        if self.seasonality != 1 and self.seasonal_period is not None:
            msg = ('seasonal_period is only valid when seasonality is 1!')
            logger.error(msg)
            raise ValueError(msg)

        if self.seasonality != 2 and self.seasonal_handle_method is not None:
            msg = ('seasonal_handle_method is only valid when seasonality is 2!')
            logger.error(msg)
            raise ValueError(msg)
        self.is_index_int = None
        self.forecast_start = None
        self.timedelta = None

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
        conn = data.connection_context
        require_pal_usable(conn)
        endog = self._arg('endog', endog, str)

        cols = data.columns
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

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['FORECAST', 'STATISTICS']
        outputs = ['#PAL_LR_SEASONAL_ADJUST_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        forecast_tbl, stats_tbl = outputs

        param_rows = [
            ('FORECAST_LENGTH', self.forecast_length, None, None),
            ('TREND', None, self.trend, None),
            ('AFFECT_FUTURE_ONLY', self.affect_future_only, None, None),
            ('SEASONALITY', self.seasonality, None, None),
            ('PERIODS', self.seasonal_period, None, None),
            ('SEASONAL_HANDLE_METHOD', self.seasonal_handle_method, None, None),
            ('IGNORE_ZERO', self.ignore_zero, None, None),
            ('EXPOST_FLAG', self.expost_flag, None, None)
        ]

        if self.accuracy_measure is not None:
            if isinstance(self.accuracy_measure, str):
                self.accuracy_measure = [self.accuracy_measure]
            for acc_measure in self.accuracy_measure:
                param_rows.extend([('MEASURE_NAME', None, None, acc_measure)])

        try:
            self._call_pal_auto(conn,
                                "PAL_LR_SEASONAL_ADJUST",
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
            fct_ = conn.sql("""
                            SELECT ADD_SECONDS('{0}', ({1}-{5}) * {2}) AS {1},
                            {4} FROM ({3})
                            """.format(self.forecast_start,
                                       quotename(self.forecast_.columns[0]),
                                       self.timedelta,
                                       self.forecast_.select_statement,
                                       quotename(self.forecast_.columns[1]),
                                       data.count() + 1))
            self.forecast_ = fct_
        return self.forecast_
