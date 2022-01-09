"""
This module contains Python wrapper for SAP HANA PAL Unified Exponential Smoothing.

The following classes are available:
    * :class:`UnifiedExponentialSmoothing`
"""
#pylint: disable=too-many-lines, too-many-branches, unused-argument
#pylint: disable=line-too-long, too-many-statements
#pylint: disable=too-many-locals, too-many-instance-attributes
#pylint: disable=too-many-arguments, invalid-name
#pylint: disable=ungrouped-imports, bare-except
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
from hana_ml.ml_base import try_drop
from hana_ml.dataframe import quotename
from hana_ml.algorithms.pal.tsa.arima import _convert_index_from_timestamp_to_int, _is_index_int
from hana_ml.algorithms.pal.tsa.arima import _get_forecast_starttime_and_timedelta
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.algorithms.pal.pal_base import (
    arg,
    PALBase,
    ParameterTable,
    require_pal_usable,
    pal_param_register,
    ListOfTuples,
    ListOfStrings)
logger = logging.getLogger(__name__)

def _acc_measure_check(func, input_dict):
    if input_dict.get('accuracy_measure') is not None:
        ac = input_dict.get('accuracy_measure')
        if isinstance(ac, str):
            ac = [ac]
        if func in ['AESM']:
            if len(ac) != 1:
                msg = "Please input accuracy_measure from 'mse' OR 'mape'!"
                logger.error(msg)
                raise ValueError(msg)
            arg('accuracy_measure', ac[0].lower(), {'mse':'mse', 'mape':'mape'})
        acc_list = {"mpe":"mpe", "mse":"mse", "rmse":"rmse", "et":"et",
                                 "mad":"mad", "mase":"mase", "wmape":"wmape",
                                 "smape":"smape", "mape":"mape"}
        if func not in ['AESM']:
            for acc in ac:
                acc = acc.lower()
                arg('accuracy_measure', acc, acc_list)


def _season_start_check(input_dict):
    if input_dict.get('season_start') is not None:
        input_dict['season_start'] = arg('season_start', input_dict['season_start'], ListOfTuples)
        for element in input_dict['season_start']:
            if len(element) != 2:
                msg = 'The length of each tuple of season_start should be 2!'
                logger.error(msg)
                raise ValueError(msg)
            if not isinstance(element[0], int):
                msg = 'The type of the first element of the tuple of season_start should be int!'
                logger.error(msg)
                raise ValueError(msg)
            if not isinstance(element[1], (float, int)):
                msg = 'The type of the second element of the tuple of season_start should be float!'
                logger.error(msg)
                raise ValueError(msg)

def _seasonal_check(input_dict):
    if input_dict.get('seasonal') is not None:
        input_dict['seasonal'] = arg('seasonal', input_dict['seasonal'], (int, str))
        if isinstance(input_dict['seasonal'], str):
            input_dict['seasonal'] = arg('seasonal', input_dict['seasonal'],
                                         dict(multiplicative=0, additive=1))

def _delta_check(input_dict):
    if input_dict.get('delta') is not None and input_dict.get('adaptive_method') is False:
        msg = ('delta is only valid when adaptive_method is True!')
        logger.error(msg)
        raise ValueError(msg)

def _other_params_check(input_dict, func_map, func):
    update_params = {}
    for parm in input_dict:
        if parm in func_map.keys():
            if parm in ['season_start', 'accuracy_measure', 'seasonal']:
                pass
            else:
                parm_val = input_dict[parm]
                arg_map = func_map[parm]
                if arg_map[1] == ListOfStrings and isinstance(parm_val, str):
                    parm_val = [parm_val]
                if len(arg_map) == 2:
                    update_params[arg_map[0]] = (arg(parm, parm_val, arg_map[1]), arg_map[1])
                else:
                    update_params[arg_map[0]] = (arg(parm, parm_val, arg_map[2]), arg_map[1])
        else:
            err_msg = "'{}' is not a valid parameter name for initializing a {} model".format(parm, func)
            logger.error(err_msg)
            raise KeyError(err_msg)

    return update_params

class UnifiedExponentialSmoothing(PALBase):
    """
    The Python wrapper for SAP HANA PAL Unified Exponential Smoothing function.

    The Unified Exponential Smoothing algorithms include:

    - SESM (Single Exponential Smoothing)
    - DESM (Double Exponential Smoothing)
    - TESM (Triple Exponential Smoothing)
    - BESM (Brown Exponential Smoothing)
    - AESM (Auto Exponential Smoothing)

    Parameters
    ----------

    func : str

        The name of a specified exponential smoothing algorithm.

        The following algorithms are supported:

        - 'SESM' : Single Exponential Smoothing.
        - 'DESM' : Double Exponential Smoothing.
        - 'TESM' : Triple Exponential Smoothing.
        - 'BESM' : Brown Exponential Smoothing.
        - 'AESM' : Auto Exponential Smoothing.

    massive : bool, optional
        Specifies whether or not to use massive mode of exponential smoothing.

        Defaults to False.

    group_params : dict, optional
        If massive mode is activated, input data for exponential smoothing shall be divided into different
        groups with different exponential smoothing parameters applied. This parameter specifies the parameter
        values of the chosen exponential smoothing algorithm ``func`` w.r.t. different groups in a dict format,
        where keys corresponding to group ids while values should be a dict for exponential smoothing algorithm
        parameter value assignments.

        Valid only when ``massive`` is True.

    **kwargs : keyword arguments

        Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.

          - **'SESM'** : :class:`~from hana_ml.algorithms.pal.tsa.exponential_smoothing.SingleExponentialSmoothing`

          - **'DESM'** : :class:`~from hana_ml.algorithms.pal.tsa.exponential_smoothing.DoubleExponentialSmoothing`

          - **'TESM'** : :class:`~from hana_ml.algorithms.pal.tsa.exponential_smoothing.TripleExponentialSmoothing`

          - **'AESM'** : :class:`~from hana_ml.algorithms.pal.tsa.exponential_smoothing.AutoExponentialSmoothing`

          - **'BESM'** : :class:`~from hana_ml.algorithms.pal.tsa.exponential_smoothing.BrownExponentialSmoothing`

        Valid only when ``massive`` is False.

    Attributes
    ----------

    forecast_ : DataFrame
        Forecast values.

    stats_ : DataFrame
        Statistics analysis content.

    error_msg_ : DataFrame

        Provides error message.


    Examples
    --------
    Input dataframe df:

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

    Create a UnifiedExponentialSmoothing instance:

    >>> ub = UnifiedExponentialSmoothing(func='besm',
                                         alpha=0.1,
                                         forecast_num=6,
                                         adaptive_method=False,
                                         accuracy_measure='mse',
                                         expost_flag=True,
                                         prediction_confidence_1=0.8,
                                         prediction_confidence_2=0.95)

    Perform fit on the given data:

    >>> ub.fit_predict(data=df)

    Output:

    >>> ub.forecast_.collect().set_index('TIMESTAMP').head(6)
      TIMESTAMP      VALUE
    0         2  143.00000
    1         3  144.80000
    2         4  148.13000
    3         5  146.55600
    4         6  144.80550
    5         7  150.70954

    >>> ub.stats_.collect()
        STAT_NAME       STAT_VALUE
    0         MSE       474.142004

    """

    func_dict = {'sesm' : 'SESM',
                 'desm' : 'DESM',
                 'tesm' : 'TESM',
                 'besm' : 'BESM',
                 'aesm' : 'AESM'}

    _sesm_param = {'alpha': ('ALPHA', float),
                   'delta': ('BETA', float),
                   'forecast_num': ('FORECAST_NUM', int),
                   'accuracy_measure': ('ACCURACY_MEASURE', str),
                   'adaptive_method': ('ADAPTIVE_METHOD', bool),
                   'ignore_zero': ('IGNORE_ZERO', bool),
                   'expost_flag': ('EXPOST_FLAG', bool),
                   'prediction_confidence_1': ('PREDICTION_CONFIDENCE_1', float),
                   'prediction_confidence_2': ('PREDICTION_CONFIDENCE_2', float)
                   }

    _desm_param = {'alpha': ('ALPHA', float),
                   'beta': ('BETA', float),
                   'forecast_num': ('FORECAST_NUM', int),
                   'accuracy_measure': ('ACCURACY_MEASURE', str),
                   'phi': ('PHI', float),
                   'damped': ('DAMPED', bool),
                   'ignore_zero': ('IGNORE_ZERO', bool),
                   'expost_flag': ('EXPOST_FLAG', bool),
                   'prediction_confidence_1': ('PREDICTION_CONFIDENCE_1', float),
                   'prediction_confidence_2': ('PREDICTION_CONFIDENCE_2', float)
                   }

    _tesm_param = {'alpha': ('ALPHA', float),
                   'beta': ('BETA', float),
                   'gamma': ('GAMMA', float),
                   'seasonal_period': ('CYCLE', int),
                   'forecast_num': ('FORECAST_NUM', int),
                   'accuracy_measure': ('ACCURACY_MEASURE', str),
                   'seasonal': ('SEASONAL', (int, str)),
                   'initial_method': ('INITIAL_METHOD', int),
                   'phi': ('PHI', float),
                   'damped': ('DAMPED', bool),
                   'ignore_zero': ('IGNORE_ZERO', bool),
                   'expost_flag': ('EXPOST_FLAG', bool),
                   'level_start': ('LEVEL_START', float),
                   'trend_start': ('TREND_START', float),
                   'season_start': ('SEASON_START', list),
                   'prediction_confidence_1': ('PREDICTION_CONFIDENCE_1', float),
                   'prediction_confidence_2': ('PREDICTION_CONFIDENCE_2', float)
                   }

    _aesm_param = {'model_selection': ('MODELSELECTION', bool),
                   'forecast_model_name': ('FORECAST_MODEL_NAME', str),
                   'optimizer_time_budget': ('OPTIMIZER_TIME_BUDGET', int),
                   'max_iter': ('MAX_ITERATION', int),
                   'optimizer_random_seed': ('OPTIMIZER_RANDOM_SEED', int),
                   'thread_ratio': ('THREAD_RATIO', float),
                   'alpha': ('ALPHA', float),
                   'beta': ('BETA', float),
                   'gamma': ('GAMMA', float),
                   'phi': ('PHI', float),
                   'forecast_num': ('FORECAST_NUM', int),
                   'accuracy_measure': ('ACCURACY_MEASURE', str),
                   'seasonal_period': ('CYCLE', int),
                   'seasonal': ('SEASONAL', (int, str)),
                   'initial_method': ('INITIAL_METHOD', int),
                   'training_ratio': ('TRAINING_RATIO', float),
                   'damped': ('DAMPED', bool),
                   'seasonality_criterion': ('SEASONALITY_CRITERION', float),
                   'trend_test_method': ('TREND_TEST_METHOD', (int, str)),
                   'trend_test_alpha': ('TREND_TEST_ALPHA', float),
                   'alpha_min': ('ALPHA_MIN', float),
                   'beta_min': ('BETA_MIN', float),
                   'gamma_min': ('GAMMA_MIN', float),
                   'phi_min': ('PHI_MIN', float),
                   'alpha_max': ('ALPHA_MAX', float),
                   'beta_max': ('BETA_MAX', float),
                   'gamma_max': ('GAMMA_MAX', float),
                   'phi_max': ('PHI_MAX', float),
                   'prediction_confidence_1': ('PREDICTION_CONFIDENCE_1', float),
                   'prediction_confidence_2': ('PREDICTION_CONFIDENCE_2', float),
                   'level_start': ('LEVEL_START', float),
                   'trend_start': ('TREND_START', float),
                   'season_start': ('SEASON_START', list)
                   }

    _besm_param = {'alpha': ('ALPHA', float),
                   'delta': ('BETA', float),
                   'forecast_num': ('FORECAST_NUM', int),
                   'accuracy_measure': ('ACCURACY_MEASURE', str),
                   'adaptive_method': ('ADAPTIVE_METHOD', bool),
                   'ignore_zero': ('IGNORE_ZERO', bool),
                   'expost_flag': ('EXPOST_FLAG', bool),
                   'prediction_confidence_1': ('PREDICTION_CONFIDENCE_1', float),
                   'prediction_confidence_2': ('PREDICTION_CONFIDENCE_2', float)
                   }

    map_dict = {'SESM' : _sesm_param,
                'DESM' : _desm_param,
                'TESM' : _tesm_param,
                'BESM' : _besm_param,
                'AESM' : _aesm_param}

    def __init__(self,
                 func,
                 massive=False,
                 group_params=None,
                 **kwargs):
        super(UnifiedExponentialSmoothing, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        func = func.lower()
        self.func = self._arg('Function name', func, self.func_dict)
        self.__real_func = self.func if massive is False else "M" + self.func
        massive = self._arg('massive', massive, bool)
        group_params = self._arg('group_params', group_params, dict)
        group_params = {} if group_params is None else group_params
        if len(group_params) > 0:
            for group in group_params:
                self._arg(self.func + ' Parameters with group_id ' + str(group),
                          group_params[group], dict)
        func_map = self.map_dict[self.func]
        self.__pal_params = {}
        if massive is False:
            self.params = dict(**kwargs)
            _acc_measure_check(func=self.func, input_dict=self.params)
            _season_start_check(input_dict=self.params)
            _seasonal_check(input_dict=self.params)
            _delta_check(input_dict=self.params)
            self.__pal_params = _other_params_check(input_dict=self.params, func_map=func_map, func=self.func)
        else:
            self.params = group_params
            for group in self.params:
                self.__pal_params[group] = {}
                _acc_measure_check(func=self.func, input_dict=self.params[group])
                _season_start_check(input_dict=self.params[group])
                _seasonal_check(input_dict=self.params[group])
                _delta_check(input_dict=self.params[group])
                self.__pal_params[group] = _other_params_check(input_dict=self.params[group],
                                                               func_map=func_map,
                                                               func=self.func)

        self.forecast_ = None
        self.stats_ = None
        self.err_msg_ = None
        self.is_index_int = None
        self.forecast_start = None
        self.timedelta = None

    def __map_param(self, name, value, typ):#pylint:disable=no-self-use
        tpl = ()
        if typ in [int, bool]:
            tpl = (name, value, None, None)
        elif typ == float:
            tpl = (name, None, value, None)
        elif typ in [str, ListOfStrings]:
            tpl = (name, None, None, value)
        elif isinstance(typ, dict):
            val = value
            if isinstance(val, (int, float)):
                tpl = (name, val, None, None)
            else:
                tpl = (name, None, None, val)
        return tpl

    @trace_sql
    def fit_predict(self,
                    data,
                    key=None,
                    endog=None,
                    group_key=None):
        """
        Fit function for unified exponential smoothing.

        Parameters
        ----------
        data : DataFrame
            Training data.

        key : str, optional

            Name of ID column.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional
            The column of series to be fitted and predicted.

            Defaults to the first column of data after eliminating key column.

        Returns
        -------

        A fitted object of ‘UnifiedExponentialSmoothing’.

        """
        conn = data.connection_context
        require_pal_usable(conn)

        cols = data.columns
        group_key_type = None
        self.is_index_int = True
        expect_len = 3 if self.__real_func[0] == "M" else 2
        if len(cols) < expect_len:
            msg = ("Input data should contain at least 2 columns: " +\
                   "" if expect_len == 2 else "group_key, " +\
                   "key and endog.")
            logger.error(msg)
            raise ValueError(msg)
        if expect_len == 3:
            group_key = self._arg('group_key', group_key, str, required=True)
            if group_key is not None and group_key not in cols:
                msg = ("Please select group_key from from '{}'!".format(cols))
                logger.error(msg)
                raise ValueError(msg)
            data_groups = list(data[[group_key]].collect()[group_key].drop_duplicates())
            param_keys = list(self.params.keys())
            gid_type = data[[group_key]].dtypes()[0]
            if not all([(int(ky) if 'INT' in gid_type[1] else ky) in data_groups for ky in param_keys]):
                msg = 'Invalid group key identified in group parameters.'
                logger.error(msg)
                raise ValueError(msg)
            if 'INT' in gid_type[1]:
                group_key_type = gid_type[1]
            elif 'VARCHAR' in gid_type[1]:
                group_key_type = gid_type[1] + '({})'.format(gid_type[2])
            cols.remove(group_key)
        key = self._arg('key', key, str)
        if key is not None and key not in cols:
            msg = ("Please select key from from '{}'!".format(cols))
            logger.error(msg)
            raise ValueError(msg)
        if self.__real_func[0] != "M":
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
            if endog is not None and endog not in cols:
                msg = ("Please select endog from from '{}'!".format(cols))
                logger.error(msg)
                raise ValueError(msg)
            if endog is None:
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

            param_rows = [('FUNCTION', None, None, self.func)]
            for name in self.__pal_params:
                value, typ = self.__pal_params[name]
                tpl = [self.__map_param(name, value, typ)]
                param_rows.extend(tpl)

            if self.params.get('accuracy_measure') is not None:
                ac = self.params.get('accuracy_measure')
                if isinstance(self.params.get('accuracy_measure'), str):
                    ac = [ac]
                for acc in ac:
                    param_rows.extend([('ACCURACY_MEASURE', None, None, acc)])
                    param_rows.extend([('MEASURE_NAME', None, None, acc)])
            if self.params.get('season_start') is not None:
                param_rows.extend([('SEASON_START', element[0], element[1], None)
                                   for element in self.params.get('season_start')])

            if self.params.get('seasonal') is not None:
                seasonal_dict = {'multiplicative':0, 'additive':1}
                if isinstance(self.params['seasonal'], str):
                    seasonal_str = self.params['seasonal']
                    seasonal_int = seasonal_dict[seasonal_str]
                if isinstance(self.params['seasonal'], int):
                    seasonal_int = self.params['seasonal']
                param_rows.extend([('SEASONAL', seasonal_int, None, None)])
        else:
            key = cols[0] if key is None else key
            cols.remove(key)
            endog = self._arg('endog', endog, str)
            if endog is not None and endog not in cols:
                msg = ("Please select endog from from '{}'!".format(cols))
                logger.error(msg)
                raise ValueError(msg)
            if endog is None:
                endog = cols[0]
            data_ = data[[group_key, key, endog]]
            self.is_index_int = _is_index_int(data_, key)
            if not self.is_index_int:
                data_ = data[[key, group_key, endog]]
                recomb_data = None
                self.forecast_start = {}
                self.timedelta = {}
                group_count = {}
                for group in data_groups:
                    group_val = group if 'INT' in group_key_type else "'{}'".format(group)
                    group_data = data_.filter("{}={}".format(quotename(data_.dtypes()[1][0]),
                                                             group_val)).sort(data_.dtypes()[1][0])
                    group_count[group] = group_data.count()
                    try:
                        self.forecast_start[group], self.timedelta[group] =\
                        _get_forecast_starttime_and_timedelta(group_data,
                                                              key,
                                                              self.is_index_int)
                    except Exception as err:
                        logger.warning(err)
                        pass
                    group_data = _convert_index_from_timestamp_to_int(group_data, key)
                    if recomb_data is None:
                        recomb_data = group_data
                    else:
                        recomb_data = recomb_data.union(group_data)
                data_ = recomb_data[[group_key, key + '(INT)', endog]]

            param_rows = [(None, 'FUNCTION', None, None, self.__real_func)]
            for group in self.__pal_params:
                group_val = int(group) if 'INT' in group_key_type else group
                for name in self.__pal_params[group]:
                    value, typ = self.__pal_params[group][name]
                    tpl = [tuple([group_val] + list(self.__map_param(name, value, typ)))]
                    param_rows.extend(tpl)

                if self.params[group].get('accuracy_measure') is not None:
                    ac = self.params[group].get('accuracy_measure')
                    if isinstance(self.params[group].get('accuracy_measure'), str):
                        ac = [ac]
                    for acc in ac:
                        param_rows.extend([(group_val, 'ACCURACY_MEASURE', None, None, acc)])
                        param_rows.extend([(group_val, 'MEASURE_NAME', None, None, acc)])
                if self.params[group].get('season_start') is not None:
                    param_rows.extend([(group_val, 'SEASON_START', element[0], element[1], None)
                                       for element in self.params[group].get('season_start')])

                if self.params[group].get('seasonal') is not None:
                    seasonal_dict = {'multiplicative':0, 'additive':1}
                    if isinstance(self.params[group]['seasonal'], str):
                        seasonal_str = self.params['seasonal']
                        seasonal_int = seasonal_dict[seasonal_str]
                    if isinstance(self.params[group]['seasonal'], int):
                        seasonal_int = self.params[group]['seasonal']
                    param_rows.extend([(group_val, 'SEASONAL', seasonal_int, None, None)])

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['FORECAST', 'STATS', 'ERROR_MSG', 'PLACE_HOLDER1', 'PLACE_HOLDER2']
        outputs = ['#PAL_UNIFIED_ES_{}_{}_{}'.format(tbl, self.id, unique_id)
                   for tbl in outputs]
        forecast_tbl, stats_tbl, error_msg_tbl, _, _ = outputs
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_EXPONENTIALSMOOTHING',
                                data_,
                                ParameterTable(itype=group_key_type).with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            logger.error(str(db_err))
            try_drop(conn, outputs)
            raise
        except pyodbc.Error as db_err:
            logger.error(str(db_err.args[1]))
            try_drop(conn, outputs)
            raise

        #pylint: disable=attribute-defined-outside-init
        self.forecast_ = conn.table(forecast_tbl)
        self.stats_ = conn.table(stats_tbl)
        self.error_msg_ = conn.table(error_msg_tbl)

        if not self.error_msg_.collect().empty:
            row = self.error_msg_.count()
            for i in range(1, row+1):
                warn_msg = "For group_key '{}',".format(self.error_msg_.collect()['GROUP_ID'][i-1]) +\
                           " the error message is '{}'.".format(self.error_msg_.collect()['MESSAGE'][i-1]) +\
                           "More information could be seen in the attribute error_msg_!"
                warnings.warn(message=warn_msg)

        if not self.is_index_int:
            if self.__real_func[0] != 'M':
                if self.func != 'BESM':
                    fct_ = conn.sql("""
                                    SELECT {0},
                                    ADD_SECONDS('{1}', ({2}-{10}) * {3}) AS {2},
                                    {5},
                                    {6},
                                    {7},
                                    {8},
                                    {9}
                                    FROM ({4})
                                    """.format(quotename(self.forecast_.columns[0]),
                                               self.forecast_start,
                                               quotename(self.forecast_.columns[1]),
                                               self.timedelta,
                                               self.forecast_.select_statement,
                                               quotename(self.forecast_.columns[2]),
                                               quotename(self.forecast_.columns[3]),
                                               quotename(self.forecast_.columns[4]),
                                               quotename(self.forecast_.columns[5]),
                                               quotename(self.forecast_.columns[6]),
                                               data.count() + 1))
                else:
                    fct_ = conn.sql("""
                                    SELECT {0},
                                    ADD_SECONDS('{1}', ({2}-{10}) * {3}) AS {2},
                                    {5},
                                    {6},
                                    {7},
                                    {8},
                                    {9}
                                    FROM ({4})
                                    """.format(quotename(self.forecast_.columns[0]),
                                               self.forecast_start,
                                               quotename(self.forecast_.columns[1]),
                                               self.timedelta,
                                               self.forecast_.select_statement,
                                               quotename(self.forecast_.columns[2]),
                                               quotename(self.forecast_.columns[3]),
                                               quotename(self.forecast_.columns[4]),
                                               quotename(self.forecast_.columns[5]),
                                               quotename(self.forecast_.columns[6]),
                                               data.count() + 1))
            else:
                comb_data = None
                fct = self.forecast_
                for group in data_groups:
                    group_val = int(group) if 'INT' in group_key_type else "'{}'".format(group)
                    group_fct = fct.filter('GROUP_ID={}'.format(group_val)).sort('TIMESTAMP')
                    if self.func != 'BESM':
                        group_fct = conn.sql("""
                                             SELECT {0},
                                             ADD_SECONDS('{1}', ({2}-{10}) * {3}) AS {2},
                                             {5},
                                             {6},
                                             {7},
                                             {8},
                                             {9}
                                             FROM ({4})
                                             """.format(quotename(self.forecast_.columns[0]),
                                                        self.forecast_start[group],
                                                        quotename(self.forecast_.columns[1]),
                                                        self.timedelta[group],
                                                        group_fct.select_statement,
                                                        quotename(self.forecast_.columns[2]),
                                                        quotename(self.forecast_.columns[3]),
                                                        quotename(self.forecast_.columns[4]),
                                                        quotename(self.forecast_.columns[5]),
                                                        quotename(self.forecast_.columns[6]),
                                                        group_count[group] + 1))
                    else:
                        group_fct = conn.sql("""
                                             SELECT {0},
                                             ADD_SECONDS('{1}', ({2}-{10}) * {3}) AS {2},
                                             {5},
                                             {6},
                                             {7},
                                             {8},
                                             {9}
                                             FROM ({4})
                                             """.format(quotename(self.forecast_.columns[0]),
                                                        self.forecast_start[group],
                                                        quotename(self.forecast_.columns[1]),
                                                        self.timedelta[group],
                                                        group_fct.select_statement,
                                                        quotename(self.forecast_.columns[2]),
                                                        quotename(self.forecast_.columns[3]),
                                                        quotename(self.forecast_.columns[4]),
                                                        quotename(self.forecast_.columns[5]),
                                                        quotename(self.forecast_.columns[6]),
                                                        group_count[group] + 1))
                    if comb_data is None:
                        comb_data = group_fct
                    else:
                        comb_data = group_fct.union(comb_data)
                fct_ = comb_data.sort(['GROUP_ID', 'TIMESTAMP'])
            self.forecast_ = fct_
        return self.forecast_
