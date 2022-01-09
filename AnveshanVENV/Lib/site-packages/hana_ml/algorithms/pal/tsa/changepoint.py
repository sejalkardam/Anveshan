"""
This module contains Python wrapper for PAL change-point detection algorithm.

The following class is available:

    * :class:`CPD`
    * :class:`BCPD`
"""
#pylint:disable=too-many-lines, line-too-long, too-many-arguments, too-few-public-methods, too-many-instance-attributes
#pylint:disable=too-many-locals, no-else-return, attribute-defined-outside-init, too-many-branches, too-many-statements
#pylint: disable=consider-using-f-string
import logging
import warnings
import uuid
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    try_drop,
    require_pal_usable
)

logger = logging.getLogger(__name__)#pylint:disable=invalid-name

class CPD(PALBase):
    r"""
    Change-point detection (CPDetection) methods aim at detecting multiple abrupt changes such as change in mean,
    variance or distribution in an observed time-series data.

    Parameters
    ----------

    cost : {'normal_mse', 'normal_rbf', 'normal_mhlb', 'normal_mv', 'linear', 'gamma', 'poisson', 'exponential', 'normal_m', 'negbinomial'}, optional

        The cost function for change-point detection.

        Defaults to 'normal_mse'.

    penalty : {'aic', 'bic', 'mbic', 'oracle', 'custom'}, optional

        The penalty function for change-point detection.

        Defaults to
            (1)'aic' if ``solver`` is 'pruneddp', 'pelt' or 'opt',

            (2)'custom' if ``solver`` is 'adppelt'.

    solver : {'pelt', 'opt', 'adppelt', 'pruneddp'}, optional

        Method for finding change-points of given data, cost and penalty.

        Each solver supports different cost and penalty functions.

          - 1.  For cost functions, 'pelt', 'opt' and 'adpelt' support the following eight:
                'normal_mse', 'normal_rbf', 'normal_mhlb', 'normal_mv',
                'linear', 'gamma', 'poisson', 'exponential';
                while 'pruneddp' supports the following four cost functions:
                'poisson', 'exponential', 'normal_m', 'negbinomial'.
          - 2.  For penalty functions, 'pruneddp' supports all penalties, 'pelt', 'opt' and 'adppelt' support the following three:
                'aic','bic','custom', while 'adppelt' only supports 'custom' cost.

        Defaults to 'pelt'.

    lamb : float, optional

        Assigned weight of the penalty w.r.t. the cost function, i.e. penalizaion factor.

        It can be seen as trade-off between speed and accuracy of running the detection algorithm.

        A small values (usually less than 0.1) will dramatically improve the efficiency.

        Defaults to 0.02, and valid only when ``solver`` is 'pelt' or 'adppelt'.

    min_size : int, optional

        The minimal length from the very begining within which change would not happen.

        Valid only when ``solver`` is 'opt', 'pelt' or 'adppelt'.

        Defaults to 2.

    min_sep : int, optional

        The minimal length of speration between consecutive change-points.

        Defaults to 1, valid only when ``solver`` is 'opt', 'pelt' or 'adppelt'.

    max_k : int, optional

        The maximum number of change-points to be detected.

        If the given value is less than 1, this number would be determined automatically from the input data.

        Defaults to 0, vaild only when ``solver`` is 'pruneddp'.

    dispersion : float, optinal

        Dispersion coefficient for Gamma and negative binomial distribution.

        Valid only when `cost` is 'gamma' or 'negbinomial'.

        Defaults to 1.0.

    lamb_range : list of two numericals(float and int) values, optional(deprecated)

        User-defined range of penalty.

        Only valid when ``solver`` is 'adppelt'.

        Deprecated, please use ``range_penalty`` instead.

    max_iter : int, optional

        Maximum number of iterations for searching the best penalty.

        Valid only when ``solver`` is 'adppelt'.

        Defaults to 40.

    range_penalty : list of two numerical values, optional

        User-defined range of penalty.

        Valid only when ``solver`` is 'adppelt' and ``value_penalty`` is not provided.

        Defaults to [0.01, 100].

    value_penalty : float, optional

        Value of user-defined penalty.

        Valid when ``penalty`` is 'custom' or ``solver`` is 'adppelt'.

        No default value.

    Attributes
    ----------

    stats_ : DataFrame

         Statistics for running change-point detection on the input data, structured as follows:
            - 1st column: statistics name,
            - 2nd column: statistics value.

    Examples
    --------

    First check the input time-series DataFrame df:

    >>> df.collect()
      TIME_STAMP      SERIES
    0        1-1       -5.36
    1        1-2       -5.14
    2        1-3       -4.94
    3        2-1       -5.15
    4        2-2       -4.95
    5        2-3        0.55
    6        2-4        0.88
    7        3-1        0.95
    8        3-2        0.68
    9        3-3        0.86

    Now create a CPD instance with 'pelt' solver and 'aic' penalty:

    >>> cpd = CPD(solver='pelt',
    ...           cost='normal_mse',
    ...           penalty='aic',
    ...           lamb=0.02)

    Apply the above CPD instance to the input data, check the detection result and
    related statistics:

    >>> cp = cpd.fit_predict(data=df)
    >>> cp.collect()
          TIME_STAMP
    0            2-2

    >>> cpd.stats_.collect()
                 STAT_NAME    STAT_VAL
    0               solver        Pelt
    1        cost function  Normal_MSE
    2         penalty type         AIC
    3           total loss     4.13618
    4  penalisation factor        0.02

    Create another CPD instance with 'adppelt' solver and 'normal_mv' cost:

    >>> cpd = CPD(solver='adppelt',
    ...           cost='normal_mv',
    ...           range_penalty=[0.01, 100],
    ...           lamb=0.02)

    Again, apply the above CPD instance to the input data, check the detection result and
    related statistics:

    >>> cp.collect()
               TIME_STAMP
         0            2-2

    >>> cpd.stats_.collect()
                 STAT_NAME   STAT_VAL
    0               solver    AdpPelt
    1        cost function  Normal_MV
    2         penalty type     Custom
    3           total loss   -28.1656
    4  penalisation factor       0.02
    5            iteration          2
    6      optimal penalty    2.50974

    Create a third CPD instance with 'pruneddp' solver and 'oracle' penalty:

    >>> cpd = CPD(solver='pruneddp', cost='normal_m', penalty='oracle', max_k=3)

    Simiar as before, apply the above CPD instance to the input data, check the detection result and
    related statistics:

    >>> cp = cpd.fit_predict(data=df)
    >>> cp.collect()
          TIME_STAMP
    0            2-2

    >>> cpd.stats_.collect()
                 STAT_NAME   STAT_VAL
    0               solver    AdpPelt
    1        cost function  Normal_MV
    2         penalty type     Custom
    3           total loss   -28.1656
    4  penalisation factor       0.02
    5            iteration          2
    6      optimal penalty    2.50974
    """

    solver_map = {'pelt':'Pelt', 'opt':'Opt', 'adppelt':'AdpPelt', 'pruneddp':'PrunedDP'}
    penalty_map = {'aic':'AIC', 'bic':'BIC', 'mbic':'mBIC', 'oracle':'Oracle', 'custom':'Custom'}
    cost_map = {'normal_mse':'Normal_MSE', 'normal_rbf':'Normal_RBF',
                'normal_mhlb':'Normal_MHLB', 'normal_mv':'Normal_MV',
                'linear':'Linear', 'gamma':'Gamma', 'poisson':'Poisson',
                'exponential':'Exponential', 'normal_m':'Normal_M',
                'negbinomial':'NegBinomial'}
    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches
                 cost=None,
                 penalty=None,
                 solver=None,
                 lamb=None,
                 min_size=None,
                 min_sep=None,
                 max_k=None,
                 dispersion=None,
                 lamb_range=None,
                 max_iter=None,
                 range_penalty=None,
                 value_penalty=None):
        super(CPD, self).__init__()
        self.cost = self._arg('cost', cost, self.cost_map)
        self.penalty = self._arg('penalty', penalty, self.penalty_map)
        self.solver = self._arg('solver', solver, self.solver_map)
        if self.solver in ('Pelt', 'Opt', None) and self.penalty not in ('AIC', 'BIC', 'Custom', None):
            msg = ("When 'solver' is 'pelt' or 'opt', "+
                   "only 'aic', 'bic' and 'custom' are valid penalty functions.")
            raise ValueError(msg)
        if self.solver == 'AdpPelt' and self.penalty not in ('Custom', None):
            msg = ("When 'solver' is 'adppelt', penalty function must be 'custom'.")
            raise ValueError(msg)
        cost_list_one = ['Normal_MSE', 'Normal_RBF', 'Normal_MHLB', 'Normal_MV',
                         'Linear', 'Gamma', 'Poisson', 'Exponential']
        cost_list_two = ['Poisson', 'Exponential', 'Normal_M', 'NegBinomial']
        if self.solver in ('Pelt', 'Opt', 'AdpPelt', None):
            if  self.cost is not None and self.cost not in cost_list_one:
                msg = ("'solver' is currently one of the following: pelt, opt and adppelt, "+
                       "in this case cost function must be one of the following: normal_mse, normal_rbf, "+
                       "normal_mhlb, normal_mv, linear, gamma, poisson, exponential.")
                raise ValueError(msg)
        elif self.cost is not None and self.cost not in cost_list_two:
            msg = ("'solver' is currently PrunedDP, in this case 'cost' must be assigned a valid value listed as follows: poisson, exponential, normal_m, negbinomial")
            raise ValueError(msg)
        self.lamb = self._arg('lamb', lamb, float)
        self.min_size = self._arg('min_size', min_size, int)
        self.min_sep = self._arg('min_sep', min_sep, int)
        self.max_k = self._arg('max_k', max_k, int)
        self.dispersion = self._arg('dispersion', dispersion, float)
        if lamb_range is not None:
            if isinstance(lamb_range, list) and len(lamb_range) == 2 and all(isinstance(val, (int, float)) for val in lamb_range):#pylint:disable=line-too-long
                self.lamb_range = lamb_range
            else:
                msg = ("Wrong setting for parameter 'lamb_range', correct setting "+
                       "should be a list of two numerical values that corresponds to "+
                       "lower- and upper-limit of the penelty weight.")
                raise ValueError(msg)
        else:
            self.lamb_range = None
        self.max_iter = self._arg('max_iter', max_iter, int)
        if range_penalty is not None:
            if isinstance(range_penalty, (list, tuple)) and len(range_penalty) == 2 and all(isinstance(val, (int, float)) for val in range_penalty):#pylint:disable=line-too-long
                self.lamb_range = list(range_penalty)
            else:
                msg = ("Wrong setting for parameter 'range_penalty', correct setting "+
                       "should be a list of two numerical values that corresponds to "+
                       "lower- and upper-limit of the penelty value.")
                raise ValueError(msg)
        else:
            self.lamb_range = None
        self.value_penalty = self._arg('value_penalty', value_penalty, float)

    def fit_predict(self, data, key=None, features=None):
        """
        Detecting change-points of the input data.

        Parameters
        ----------

        data : DataFrame

            Input time-series data for change-point detection.

        key : str, optional

            Column name for time-stamp of the input time-series data.

            If the index column of data is not provided or not a single column, and the key of fit_predict function is not provided,
            the default value is the first column of data.

            If the index of data is set as a single column, the default value of key is index column of data.

        features : str or list of str, optional

            Column name(s) for the value(s) of the input time-series data.

        Returns
        -------

        DataFrame

            Detected the change-points of the input time-series data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
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

        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = self._arg('features', features, ListOfStrings)
            except:
                msg = ("'features' must be list of string or string.")
                raise TypeError(msg)
        else:
            cols.remove(key)
            features = cols
        used_cols = [key] + features
        if any(col not in data.columns for col in used_cols):
            msg = "'key' or 'features' parameter contains unrecognized column name."
            raise ValueError(msg)
        data_ = data[used_cols]
        param_rows = [
            ('COSTFUNCTION', None, None, self.cost),
            ('SOLVER', None, None, self.solver),
            ('PENALIZATION_FACTOR', None, self.lamb, None),
            ('MIN_SIZE', self.min_size, None, None),
            ('MIN_SEP', self.min_sep, None, None),
            ('MaxK', self.max_k, None, None),
            ('DISPERSION', None, self.dispersion, None),
            ('MAX_ITERATION', self.max_iter, None, None)]
        if (self.penalty == 'Custom' or self.solver == 'AdpPelt') and self.value_penalty is not None:
            param_rows.extend([('PENALTY', None, self.value_penalty, 'Custom')])
        elif self.penalty not in ['Custom', None]:
            param_rows.extend([('PENALTY', None, None, self.penalty)])
        if self.lamb_range is not None:
            param_rows.extend([('RANGE_PENALTIES', None, None, str(self.lamb_range))])
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'STATS']
        tables = ["#PAL_CPDETECTION_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        result_tbl, stats_tbl = tables
        try:
            self._call_pal_auto(conn,
                                "PAL_CPDETECTION",
                                data_,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        self.stats_ = conn.table(stats_tbl)
        return conn.table(result_tbl)

class BCPD(PALBase):
    r"""
    Bayesian  Change-point detection (BCPD) detects abrupt changes in the time series.
    It, to some extent, can been assumed as an enhanced version of seasonality test in additive mode.

    Parameters
    ----------

    max_tcp : int

        Maximum number of trend change points to be detected.

    max_scp : int

        Maximum number of season change points to be detected.

    trend_order : int, optional

        Order of trend segments that used for decomposation

        Defaults to 1.

    max_harmonic_order : int, optional

        Maximum order of harmonic waves within seasonal segments.

        Defaults to 10.

    min_period : int, optional

        Minimum possible period within seasonal segments.

        Defaults to 1.

    max_period : int, optional

        Maximum possible period within seasonal segments.

        Defaults to half of the data length.

    random_seed : int, optional

        Indicates the seed used to initialize the random number generator:

          - 0: Uses the system time.
          - Not 0: Uses the provided value.

        Defaults to 0.

    max_iter : int, optional

        BCPD is iterative, the more iterations, the more precise will the result be rendered.

        Defaults to 5000.

    interval_ratio : float, optional

        Regulates the interval between change points, which should be larger than the corresponding portion of total length.

        Defaults to 0.1.

    Examples
    --------
    >>> df.collect()
      TIME_STAMP      SERIES
    0          1       -5.36
    1          2       -5.14
    2          3       -4.94
    3          4       -5.15
    4          5       -4.95
    5          6        0.55
    6          7        0.88
    7          8        0.95
    8          9        0.68
    9         10        0.86

    >>> bcpd = BCPD(5, 5)
    >>> tcp, scp, period, components = bcpd.fit_predict(data=df)
    >>> scp.collect()
          ID      SEASON_CP
    0      1              4
    1      2              5
    """

    def __init__(self,#pylint: disable=too-many-arguments, too-many-locals, too-many-branches
                 max_tcp,
                 max_scp,
                 trend_order=None,
                 max_harmonic_order=None,
                 min_period=None,
                 max_period=None,
                 random_seed=None,
                 max_iter=None,
                 interval_ratio=None):

        if max_scp > 0 and max_harmonic_order is None:
            warn_msg = "Please enter a positive value of max_harmonic_order when max_scp is larger than 0!"
            warnings.warn(message=warn_msg)
        super(BCPD, self).__init__()
        self.trend_order = self._arg('trend_order', trend_order, int)
        self.max_tcp = self._arg('max_tcp', max_tcp, int)
        self.max_scp = self._arg('max_scp', max_scp, int)
        self.max_harmonic_order = self._arg('max_harmonic_order', max_harmonic_order, int)
        self.min_period = self._arg('min_period', min_period, int)
        self.max_period = self._arg('max_period', max_period, int)
        self.random_seed = self._arg('random_seed', random_seed, int)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.interval_ratio = self._arg('interval_ratio', interval_ratio, float)

    def fit_predict(self, data, key=None, endog=None, features=None):
        """
        Detecting change-points of the input data.

        Parameters
        ----------

        data : DataFrame

            Input time-series data for change-point detection.

        key : str, optional

            Column name for time-stamp of the input time-series data.

            If the index column of data is not provided or not a single column, and the key of fit_predict function is not provided,
            the default value is the first column of data.

            If the index of data is set as a single column, the default value of key is index column of data.

        endog : str, optional

            Column name for the value of the input time-series data.
            Defaults to the first non-key column.

        features : str or list of str, optional (*deprecated*)
            Column name(s) for the value(s) of the input time-series data.

        Returns
        -------

        DataFrame

            The detected the trend change-points of the input time-series data.

        DataFrame

            The detected the season change-points of the input time-series data.

        DataFrame

            The detected the period within each season segment of the input time-series data.

        DataFrame

            The decomposed components.
        """
        conn = data.connection_context
        require_pal_usable(conn)
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

        if endog is not None:
            features = endog
        if features is not None:
            if not isinstance(features, str):
                msg = "BCPD currently only supports one column of endog!"
                raise ValueError(msg)
        else:
            cols.remove(key)
            features = cols[0]
        used_cols = [key] + [features]
        if any(col not in data.columns for col in used_cols):
            msg = "'key' or 'endog' parameter contains unrecognized column name."
            raise ValueError(msg)
        data_ = data[used_cols]
        param_rows = [
            ('TREND_ORDER', self.trend_order, None, None),
            ('MAX_TCP_NUM', self.max_tcp, None, None),
            ('MAX_SCP_NUM', self.max_scp, None, None),
            ('MAX_HARMONIC_ORDER', self.max_harmonic_order, None, None),
            ('MIN_PERIOD', self.min_period, None, None),
            ('MAX_PERIOD', self.max_period, None, None),
            ('RANDOM_SEED', self.random_seed, None, None),
            ('MAX_ITER', self.max_iter, None, None),
            ('INTERVAL_RATIO', None, self.interval_ratio, None)]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['TREND_CHANGE_POINT', 'SEASON_CHANGE_POINT', 'PERIOD_LIST', 'DECOMPOSED']
        tables = ["#PAL_BCPD_{}_TBL_{}_{}".format(tbl, self.id, unique_id) for tbl in tables]
        try:
            self._call_pal_auto(conn,
                                "PAL_BCPD",
                                data_,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        return (conn.table(tables[0]),
                conn.table(tables[1]),
                conn.table(tables[2]),
                conn.table(tables[3]))
