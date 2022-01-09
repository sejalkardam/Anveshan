"""
This module contains Python wrappers for PAL preprocessing algorithms.

The following classes and functions are available:

    * :class:`FeatureNormalizer`
    * :Class:`FeatureSelection`
    * :class:`KBinsDiscretizer`
    * :class:`Imputer`
    * :class:`Discretize`
    * :class:`MDS`
    * :class:`SMOTE`
    * :class:`SMOTETomek`
    * :class:`TomekLinks`
    * :class:`Sampling`
    * :func:`variance_test`
"""

#pylint: disable=line-too-long, unused-variable, raise-missing-from
#pylint: disable=consider-using-f-string
import logging
import uuid
import json
from deprecated import deprecated
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from .pal_base import (
    PALBase,
    ParameterTable,
    INTEGER,
    DOUBLE,
    NVARCHAR,
    ListOfStrings,
    ListOfTuples,
    arg,
    pal_param_register,
    require_pal_usable,
    call_pal_auto_with_hint,
    try_drop
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name
#pylint:disable=too-many-lines
class FeatureNormalizer(PALBase):#pylint: disable=too-many-instance-attributes
    """
    Normalize a DataFrame. In real world scenarios the collected continuous attributes are usually distributed within different ranges.
    It is a common practice to have the data well scaled so that data mining algorithms like neural networks,
    nearest neighbor classification and clustering can give more reliable results.

    .. Note::

       Note that the data type of the output value is the same as that of the input value.
       Therefore, if the data type of the original data is INTEGER, the output value will be converted to an integer instead of the result you expect.

       For example, if we want to use min-max method to normalize a list [1, 2, 3, 4] and set new_min = 0 and new_max = 1.0,
       we want the result to be [0, 0.33, 0.66, 1], but actually the output is [0, 0, 0, 1]
       due to the rule of consistency of input and output data type.

       Therefore, please cast the feature column(s) from INTEGER to be DOUBLE before invoking the function.

    Parameters
    ----------
    method : {'min-max', 'z-score', 'decimal'}

        Scaling methods:

            - 'min-max': Min-max normalization.
            - 'z-score': Z-Score normalization.
            - 'decimal': Decimal scaling normalization.

    z_score_method : {'mean-standard', 'mean-mean', 'median-median'}, optional

        Only valid when ``method`` is 'z-score'.

            - 'mean-standard': Mean-Standard deviation
            - 'mean-mean': Mean-Mean deviation
            - 'median-median': Median-Median absolute deviation

    new_max : float, optional

        The new maximum value for min-max normalization.

        Only valid when ``method`` is 'min-max'.

    new_min : float, optional

        The new minimum value for min-max normalization.

        Only valid when ``method`` is 'min-max'.


    thread_ratio : float, optional

        Controls the proportion of available threads to use.
        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads. Values between 0 and 1
        will use that percentage of available threads. Values outside this range
        tell PAL to heuristically determine the number of threads to use.

        Defaults to 0.

    Attributes
    ----------

    result_ : DataFrame

        Scaled dataset from fit and fit_transform methods.

    model_ :

        Trained model content.

    Examples
    --------

    Input DataFrame df1:

    >>> df1.head(4).collect()
        ID    X1    X2
    0    0   6.0   9.0
    1    1  12.1   8.3
    2    2  13.5  15.3
    3    3  15.4  18.7

    Creating a FeatureNormalizer instance:

    >>> fn = FeatureNormalizer(method="min-max", new_max=1.0, new_min=0.0)

    Performing fit on given DataFrame:

    >>> fn.fit(df1, key='ID')
    >>> fn.result_.head(4).collect()
        ID        X1        X2
    0    0  0.000000  0.033175
    1    1  0.186544  0.000000
    2    2  0.229358  0.331754
    3    3  0.287462  0.492891

    Input DataFrame for transforming:

    >>> df2.collect()
       ID  S_X1  S_X2
    0   0   6.0   9.0
    1   1   6.0   7.0
    2   2   4.0   4.0
    3   3   1.0   2.0
    4   4   9.0  -2.0
    5   5   4.0   5.0

    Performing transform on given DataFrame:

    >>> result = fn.transform(df2, key='ID')
    >>> result.collect()
       ID      S_X1      S_X2
    0   0  0.000000  0.033175
    1   1  0.000000 -0.061611
    2   2 -0.061162 -0.203791
    3   3 -0.152905 -0.298578
    4   4  0.091743 -0.488152
    5   5 -0.061162 -0.156398
    """

    method_map = {'min-max': 0, 'z-score': 1, 'decimal': 2}
    z_score_method_map = {'mean-standard': 0, 'mean-mean': 1, 'median-median': 2}

    def __init__(self,#pylint: disable=too-many-arguments
                 method,
                 z_score_method=None,
                 new_max=None,
                 new_min=None,
                 thread_ratio=None):
        setattr(self, 'hanaml_parameters', pal_param_register())
        super(FeatureNormalizer, self).__init__()
        self.method = self._arg('method', method, self.method_map, required=True)
        self.z_score_method = self._arg('z_score_method', z_score_method, self.z_score_method_map)
        self.new_max = self._arg('new_max', new_max, float)
        self.new_min = self._arg('new_min', new_min, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)

        if z_score_method is not None:
            if method.lower() != 'z-score':
                msg = 'z_score_method is not applicable when scale method is not z-score.'
                logger.error(msg)
                raise ValueError(msg)
        else:
            if method.lower() == 'z-score':
                msg = 'z_score_method must be provided when scale method is z-score.'
                logger.error(msg)
                raise ValueError(msg)

        if method.lower() == 'min-max':
            if new_min is None or new_max is None:
                msg = 'new_min and new_max must be provided when scale method is min-max.'
                logger.error(msg)
                raise ValueError(msg)

        if method.lower() != 'min-max':
            if new_min is not None or new_max is not None:
                msg = 'new_min or new_max is not applicable when scale method is not min-max.'
                logger.error(msg)
                raise ValueError(msg)

    def fit(self, data, key=None, features=None):#pylint:disable=invalid-name, too-many-locals
        """
        Normalize input data and generate a scaling model using one of the three
        scaling methods: min-max normalization, z-score normalization and
        normalization by decimal scaling.

        Parameters
        ----------

        data : DataFrame

            DataFrame to be normalized.

        key : str, optional

            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : list of str, optional

            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID columns.

        Returns
        -------

        Fitted object.
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        conn = data.connection_context
        require_pal_usable(conn)
        index = data.index
        key = self._arg('key', key, str, required=not isinstance(index, str))
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data_ = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'MODEL', 'STATISTIC', 'PLACEHOLDER']
        outputs = ['#PAL_FN_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, model_tbl, statistic_tbl, placeholder_tbl = outputs

        param_rows = [
            ('SCALING_METHOD', self.method, None, None),
            ('Z-SCORE_METHOD', self.z_score_method, None, None),
            ('THREAD_RATIO', None, self.thread_ratio, None),
            ('NEW_MAX', None, self.new_max, None),
            ('NEW_MIN', None, self.new_min, None)
            ]

        try:
            self._call_pal_auto(conn,
                                'PAL_SCALE',
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
        # pylint: disable=attribute-defined-outside-init
        self.result_ = conn.table(result_tbl)
        self.model_ = conn.table(model_tbl)
        return self

    def fit_transform(self, data, key=None, features=None):#pylint:disable=invalid-name
        """
        Fit with the dataset and return the results.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be normalized.

        key : str
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : list of str, optional
            Names of the feature columns.
            If ``features`` is not provided, it defaults to all non-ID columns.

        Returns
        -------

        DataFrame
            Normalized result, with the same structure as ``data``.
        """
        self.fit(data, key, features)
        return self.result_

    def transform(self, data, key=None, features=None):#pylint:disable=invalid-name
        """
        Scales data based on the previous scaling model.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be normalized.

        key : str, optional
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : list of str, optional
            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID columns.

        Returns
        -------

        DataFrame
            Normalized result, with the same structure as ``data``.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if self.model_ is None:
            raise FitIncompleteError("Model not initialized.")
        index = data.index
        key = self._arg('key', key, str, required=not isinstance(index, str))
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()

        result_tbl = '#PAL_FN_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        try:
            self._call_pal_auto(conn,
                                'PAL_SCALE_WITH_MODEL',
                                data_,
                                self.model_,
                                ParameterTable(),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)

class KBinsDiscretizer(PALBase):
    r"""
    Bin continuous data into number of intervals and perform local smoothing.

    .. Note::

       Note that the data type of the output value is the same as that of the input value.
       Therefore, if the data type of the original data is INTEGER, the output value will be converted to an integer instead of the result you expect.

       Therefore, please cast the feature column(s) from INTEGER to be DOUBLE before invoking the function.

    Parameters
    ----------

    strategy : {'uniform_number', 'uniform_size', 'quantile', 'sd'}
        Specifies the binning method, valid options include:

            - 'uniform_number': Equal widths based on the number of bins.
            - 'uniform_size': Equal widths based on the bin size.
            - 'quantile': Equal number of records per bin.
            - 'sd': Bins are divided based on the distance from the mean.
              Most bins are one standard deviation wide, except that the
              center bin contains all values within one standard deviation
              from the mean, and the leftmost and rightmost bins contain
              all values more than ``n_sd`` standard deviations from the
              mean in the corresponding directions.

    smoothing : {'means', 'medians', 'boundaries'}
        Specifies the smoothing method, valid options include:

            - 'means': Each value within a bin is replaced by the average of
              all the values belonging to the same bin.
            - 'medians': Each value in a bin is replaced by the median of all
              the values belonging to the same bin.
            - 'boundaries': The minimum and maximum values in a given bin are
              identified as the bin boundaries.
              Each value in the bin is then
              replaced by its closest boundary value.
              When the distance is equal to both sides, it will be replaced
              by the front boundary value.

        Values used for smoothing are not re-calculated during transform.

    n_bins : int, optional
        The number of bins.

        Only valid when ``strategy`` is 'uniform_number' or 'quantile'.

        Defaults to 2.
    bin_size : int, optional
        The interval width of each bin.

        Only valid when ``strategy`` is 'uniform_size'.

        Defaults to 10.
    n_sd : int, optional
        The leftmost bin contains all values located further than n_sd
        standard deviations lower than the mean, and the rightmost bin
        contains all values located further than n_sd standard deviations
        above the mean.

        Only valid when ``strategy`` is 'sd'.

        Defaults to 1.

    Attributes
    ----------
    result_ : DataFrame
        Binned dataset from fit and fit_transform methods.

    model_ : DataFrame
        Binning model content.

    Examples
    --------
    Input DataFrame df1:

    >>> df1.collect()
        ID  DATA
    0    0   6.0
    1    1  12.0
    2    2  13.0
    3    3  15.0
    4    4  10.0
    5    5  23.0
    6    6  24.0
    7    7  30.0
    8    8  32.0
    9    9  25.0
    10  10  38.0

    Creating a KBinsDiscretizer instance:

    >>> binning = KBinsDiscretizer(strategy='uniform_size', smoothing='means', bin_size=10)

    Performing fit on the given DataFrame:

    >>> binning.fit(data=df1, key='ID')
        ID  BIN_INDEX       DATA
    0    0          1   8.000000
    1    1          2  13.333333
    2    2          2  13.333333
    3    3          2  13.333333
    4    4          1   8.000000
    5    5          3  25.500000
    6    6          3  25.500000
    7    7          3  25.500000
    8    8          4  35.000000
    9    9          3  25.500000
    10  10          4  35.000000

    Input DataFrame df2 for transforming:

    >>> df2.collect()
       ID  DATA
    0   0   6.0
    1   1  67.0
    2   2   4.0
    3   3  12.0
    4   4  -2.0
    5   5  40.0

    Performing transform on the given DataFrame:

    >>> result = binning.transform(data=df2, key='ID')

    Output:

    >>> result.collect()
       ID  BIN_INDEX       DATA
    0   0          1   8.000000
    1   1         -1  67.000000
    2   2          1   8.000000
    3   3          2  13.333333
    4   4          1   8.000000
    5   5          4  35.000000
    """

    strategy_map = {'uniform_number': 0, 'uniform_size': 1, 'quantile': 2, 'sd': 3}
    smooth_map = {'means': 0, 'medians': 1, 'boundaries': 2}

    def __init__(self,#pylint: disable=too-many-arguments
                 strategy,
                 smoothing,
                 n_bins=None,
                 bin_size=None,
                 n_sd=None):
        super(KBinsDiscretizer, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.strategy = self._arg('strategy', strategy, self.strategy_map, required=True)
        self.smoothing = self._arg('smoothing', smoothing, self.smooth_map, required=True)
        self.n_bins = self._arg('n_bins', n_bins, int)
        self.bin_size = self._arg('bin_size', bin_size, int)
        self.n_sd = self._arg('n_sd', n_sd, int)
        #following checks are based on PAL docs, pal example has 'sd' with uniform_size
        #tested that pal ignores SD in actual executions
        if (strategy.lower() != 'uniform_number' and strategy.lower() != 'quantile'
                and n_bins is not None):
            msg = "n_bins is only applicable when strategy is uniform_number or quantile."
            logger.error(msg)
            raise ValueError(msg)
        if strategy.lower() != 'uniform_size' and bin_size is not None:
            msg = "bin_size is only applicable when strategy is uniform_size."
            logger.error(msg)
            raise ValueError(msg)
        if strategy.lower() != 'sd' and n_sd is not None:
            msg = "n_sd is only applicable when strategy is sd."
            logger.error(msg)
            raise ValueError(msg)

    def fit(self, data, key=None, features=None):#pylint: disable=too-many-locals
        """
        Bin input data into number of intervals and smooth.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be discretized.
        key : str, optional
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : list of str, optional
            Names of the feature columns.

            Since the underlying PAL bining algorithm only supports one feature,
            this list can only contain one element.

            If not provided, ``data`` must have exactly 1 non-ID column, and ``features`` defaults to that column.

        Returns
        -------

        Fitted object.
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        conn = data.connection_context
        require_pal_usable(conn)
        index = data.index
        key = self._arg('key', key, str, required=not isinstance(index, str))
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        if len(features) != 1:
            msg = ('PAL binning requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data_ = data[[key] + features]
        #PAL_BINNING requires stats and placeholder table which is not mentioned in PAL doc
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'MODEL', 'STATISTIC', 'PLACEHOLDER']
        outputs = ['#PAL_BINNING_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, model_tbl, statistic_tbl, placeholder_tbl = outputs

        param_rows = [
            ('BINNING_METHOD', self.strategy, None, None),
            ('SMOOTH_METHOD', self.smoothing, None, None),
            ('BIN_NUMBER', self.n_bins, None, None),
            ('BIN_DISTANCE', self.bin_size, None, None),
            ('SD', self.n_sd, None, None)
            ]

        try:
            self._call_pal_auto(conn,
                                'PAL_BINNING',
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
        # pylint: disable=attribute-defined-outside-init
        self.result_ = conn.table(result_tbl)
        self.model_ = conn.table(model_tbl)
        return self

    def fit_transform(self, data, key=None, features=None):#pylint:disable=invalid-name
        """
        Fit with the dataset and return the results.

        Parameters
        ----------

        data : DataFrame
            DataFrame to be binned.
        key : str, optional
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : list of str, optional
            Names of the feature columns.

            Since the underlying PAL binning algorithm only supports one feature,
            this list can only contain one element.

            If not provided, ``data`` must have exactly 1
            non-ID column, and ``features`` defaults to that column.

        Returns
        -------
        DataFrame
            Binned result, structured as follows:

              - DATA_ID column: with same name and type as ``data``'s ID column.
              - BIN_INDEX: type INTEGER, assigned bin index.
              - BINNING_DATA column: smoothed value, with same name and
                type as ``data``'s feature column.

        """

        self.fit(data, key, features)
        return self.result_

    def transform(self, data, key=None, features=None):#pylint:disable=invalid-name
        """
        Bin data based on the previous binning model.

        Parameters
        ----------
        data : DataFrame
            DataFrame to be binned.
        key : str, optional
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.
        features : list of str, optional
            Names of the feature columns.

            Since the underlying PAL_BINNING_ASSIGNMENT only supports one feature, this list can
            only contain one element.

            If not provided, ``data`` must have exactly 1 non-ID column, and ``features`` defaults to that column.

        Returns
        -------
        DataFrame
            Binned result, structured as follows:

              - DATA_ID column: with same name and type as ``data`` 's ID column.
              - BIN_INDEX: type INTEGER, assigned bin index.
              - BINNING_DATA column: smoothed value, with same name and
                type as ``data`` 's feature column.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        if self.model_ is None:
            raise FitIncompleteError("Model not initialized.")
        index = data.index
        key = self._arg('key', key, str, required=not isinstance(index, str))
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        features = self._arg('features', features, ListOfStrings)

        cols = data.columns
        cols.remove(key)
        if features is None:
            features = cols

        if len(features) != 1:
            msg = ('PAL binning assignment requires exactly one ' +
                   'feature column.')
            logger.error(msg)
            raise TypeError(msg)

        data_ = data[[key] + features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = '#PAL_BINNING_RESULT_TBL_{}_{}'.format(self.id, unique_id)

        try:
            self._call_pal_auto(conn,
                                'PAL_BINNING_ASSIGNMENT',
                                data_,
                                self.model_,
                                ParameterTable(),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)

#pylint: disable=too-many-instance-attributes, too-few-public-methods
class Imputer(PALBase):
    r"""
    Missing value imputation for DataFrame.

    Parameters
    ----------
    strategy : {'non', 'most_fequent-mean', 'most_frequent-median', 'most_frequent-zero', 'most_frequent_als', 'delete'}, optional
        Specifies the **overall** imputation strategy.

            - 'non' : No imputation for *all* columns.
            - 'most_frequent-mean' : Replacing missing values in any categorical column by its most frequently observed value, and
              missing values in any numerical column by its mean.
            - 'most_fequent-median' : Replacing missing values in any categorical column by its most frequently observed value,
              and missing values in any numerical column by its median.
            - 'most_frequent-zero' : Replacing missing values in any categorical column by its most frequently observed value, and
              missing values in all numerical columns by zeros.
            - 'most_frequent-als' : Replacing missing values in any categorical column by
              its most frequently observed value, and filling the missing values in all numerical columns via a
              matrix completion technique called **alternating least squares**.
            - 'delete' : Delete *all* rows with missing values.

        Defaults to 'most_frequent-mean'.

    thread_ratio : float, optional

        Controls the proportion of available threads to use.

        The value range is from 0 to 1, where 0 indicates a single thread,
        and 1 indicates up to all available threads.

        Values between 0 and 1 will use up to that percentage of available threads.

        Values outside this range tell PAL to heuristically determine the number of
        threads to use.

        Defaults to 0.0.

        .. note::

            The following parameters all have pre-fix 'als\_', and are invoked only when 'als' is the overall imputation strategy. \
            Those parameters are for setting up the alternating-least-square(ALS) mdoel for data imputation.

    als_factors : int, optional

        Length of factor vectors in the ALS model.

        It should be less than the number of numerical columns,
        so that the imputation results would be meaningful.

        Defaults to 3.

    als_lambda : float, optional

        L2 regularization applied to the factors in the ALS model.

        Should be non-negative.

        Defaults to 0.01.

    als_maxit : int, optional

        Maximum number of iterations for solving the ALS model.

        Defaults to 20.

    als_randomstate : int, optional

        Specifies the seed of the random number generator used
        in the training of ALS model:

            - 0: Uses the current time as the seed,
            - Others: Uses the specified value as the seed.

        Defaults to 0.

    als_exit_threshold : float, optional

        Specify a value for stopping the training of ALS nmodel.
        If the improvement of the cost function of the ALS model
        is less than this value between consecutive checks, then
        the training process will exit.

        0 means there is no checking of the objective value when
        running the algorithms, and it stops till the maximum number of
        iterations has been reached.

        Defaults to 0.

    als_exit_interval : int, optional

        Specify the number of iterations between consecutive checking of
        cost functions for the ALS model, so that one can see if the
        pre-specified ``exit_threshold`` is reached.

        Defaults to 5.

    als_linsolver : {'cholesky', 'cg'}, optional

        Linear system solver for the ALS model.

          - 'cholesky' is usually much faster.
          - 'cg' is recommended when ``als_factors`` is large.

        Defaults to 'cholesky'.

    als_maxit : int, optional

        Specifies the maximum number of iterations for cg algorithm.

        Invoked only when the 'cg' is the chosen linear system solver for ALS.

        Defaults to 3.

    als_centering : bool, optional

        Whether to center the data by column before training the ALS model.

        Defaults to True.

    als_scaling : bool, optional

        Wheter to scale the data by column before training the ALS model.

        Defaults to True.

    Attributes
    ----------

    model_ : DataFrame

        statistics/model content.

    Examples
    --------

    Input DataFrame df:

    >>> df.head(5).collect()
       V0   V1 V2   V3   V4    V5
    0  10  0.0  D  NaN  1.4  23.6
    1  20  1.0  A  0.4  1.3  21.8
    2  50  1.0  C  NaN  1.6  21.9
    3  30  NaN  B  0.8  1.7  22.6
    4  10  0.0  A  0.2  NaN   NaN

    Create an Imputer instance using 'mean' strategy and call fit:

    >>> impute = Imputer(strategy='most_frequent-mean')
    >>> result = impute.fit_transform(df, categorical_variable=['V1'],
    ...                      strategy_by_col=[('V1', 'categorical_const', '0')])

    >>> result.head(5).collect()
       V0  V1 V2        V3        V4         V5
    0  10   0  D  0.507692  1.400000  23.600000
    1  20   1  A  0.400000  1.300000  21.800000
    2  50   1  C  0.507692  1.600000  21.900000
    3  30   0  B  0.800000  1.700000  22.600000
    4  10   0  A  0.200000  1.469231  20.646154

    The stats/model content of input DataFrame:

    >>> impute.head(5).collect()
                STAT_NAME                   STAT_VALUE
    0  V0.NUMBER_OF_NULLS                            3
    1  V0.IMPUTATION_TYPE                         MEAN
    2    V0.IMPUTED_VALUE                           24
    3  V1.NUMBER_OF_NULLS                            2
    4  V1.IMPUTATION_TYPE  SPECIFIED_CATEGORICAL_VALUE

    The above stats/model content of the input DataFrame can be applied
    to imputing another DataFrame with the same data structure, e.g. consider
    the following DataFrame with missing values:

    >>> df1.collect()
       ID    V0   V1    V2   V3   V4    V5
    0   0  20.0  1.0     B  NaN  1.5  21.7
    1   1  40.0  1.0  None  0.6  1.2  24.3
    2   2   NaN  0.0     D  NaN  1.8  22.6
    3   3  50.0  NaN     C  0.7  1.1   NaN
    4   4  20.0  1.0     A  0.3  NaN  20.6

    With attribute impute being obtained, one can impute the
    missing values of df1 via the following line of code, and then check
    the result:

    >>> result1, _ = impute.transform(data=df1, key='ID')
    >>> result1.collect()
       ID  V0  V1 V2        V3        V4         V5
    0   0  20   1  B  0.507692  1.500000  21.700000
    1   1  40   1  A  0.600000  1.200000  24.300000
    2   2  24   0  D  0.507692  1.800000  22.600000
    3   3  50   0  C  0.700000  1.100000  20.646154
    4   4  20   1  A  0.300000  1.469231  20.600000

    Create an Imputer instance using other strategies, e.g. 'als' strategy
    and then call fit:

    >>> impute = Imputer(strategy='als', als_factors=2, als_randomstate=1)

    Output:

    >>> result2 = impute.fit_transform(data=df, categorical_variable=['V1'])

    >>> result2.head(5).collect()
       V0  V1 V2        V3        V4         V5
    0  10   0  D  0.306957  1.400000  23.600000
    1  20   1  A  0.400000  1.300000  21.800000
    2  50   1  C  0.930689  1.600000  21.900000
    3  30   0  B  0.800000  1.700000  22.600000
    4  10   0  A  0.200000  1.333668  21.371753

    """

    overall_imputation_map = {'non':0, 'delete': 5,
                              'most_frequent-mean':1, 'mean':1,
                              'most_frequent-median':2, 'median':2,
                              'most_frequent-zero':3, 'zero':3,
                              'most_frequent-als':4, 'als':4}
    column_imputation_map = {'non':101, 'delete':1,
                             'most_frequent':100,
                             'categorical_const':101,
                             'mean':200, 'median':201,
                             'numerical_const':203,
                             'als':204}
    dtype_escp = {'INT':INTEGER, 'DOUBLE':DOUBLE,
                  'NVARCHAR':NVARCHAR(5000), 'VARCHAR':NVARCHAR(256)}
    solver_map = {'cholsky':0, 'cg':1, 'cholesky':0}
    #pylint:disable=too-many-arguments
    def __init__(self,
                 strategy=None,
                 als_factors=None,
                 als_lambda=None,
                 als_maxit=None,
                 als_randomstate=None,
                 als_exit_threshold=None,
                 als_exit_interval=None,
                 als_linsolver=None,
                 als_cg_maxit=None,
                 als_centering=None,
                 als_scaling=None,
                 thread_ratio=None):
        setattr(self, 'hanaml_parameters', pal_param_register())
        super(Imputer, self).__init__()
        self.strategy = self._arg('strategy', strategy, self.overall_imputation_map)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.als_factors = self._arg('als_factors', als_factors, int)
        self.als_lambda = self._arg('als_lambda', als_lambda, float)
        self.als_maxit = self._arg('als_maxit', als_maxit, int)
        self.als_randomstate = self._arg('als_randomstate', als_randomstate, int)
        self.als_exit_threshold = self._arg('als_exit_threshold', als_exit_threshold,
                                            float)
        self.als_exit_interval = self._arg('als_exit_interval', als_exit_interval, int)
        self.als_linsolver = self._arg('als_linsolver', als_linsolver, self.solver_map)
        self.als_cg_maxit = self._arg('als_cg_maxit', als_cg_maxit, int)
        self.als_centering = self._arg('als_centering', als_centering, bool)
        self.als_scaling = self._arg('als_scaling', als_scaling, bool)
        self.model_ = None

    #pylint:disable=attribute-defined-outside-init, too-many-locals
    def fit_transform(self, data, key=None,#pylint:disable=too-many-branches
                      categorical_variable=None,
                      strategy_by_col=None):
        """
        Inpute the missing values of a DataFrame, return the result,
        and collect the related statistics/model info for imputation.

        Parameters
        ----------

        data : DataFrame
            Input data with missing values.
        key : str, optional
            Name of the ID column.

            If ``key`` is not provided, then:

                - if ``data`` is indexed by a single column, then ``key`` defaults
                  to that index column;
                - otherwise, it is assumed that ``data`` contains no ID column.

        categorical_variable : str, optional
            Names of columns with INTEGER data type that should actually
            be treated as categorical.

            By default, columns of INTEGER and DOUBLE type are all treated
            numerical, while columns of VARCHAR or NVARCHAR type are treated
            as categorical.
        strategy_by_col : ListOfTuples, optional
            Specifies the imputation strategy for a set of columns, which
            overrides the overall strategy for data imputation.

            Each tuple in the list should contain at least two elements,
            such that:

              - the 1st element is the name of a column;
              - the 2nd element is the imputation strategy of that column,
                valid strategies include: **'non', 'delete', 'most_frequent', 'categorical_const', \
                'mean', 'median', 'numerical_const', 'als'**.
              - If the imputation strategy is 'categorical_const' or 'numerical_const',
                then a 3rd element must be included in the tuple, which specifies
                the constant value to be used to substitute the detected missing values
                in the column.

            An example for illustration:
                [('V1', 'categorical_const', '0'),

                ('V5','median')]

        Returns
        -------

        DataFrame
            Imputed result using specified strategy, with the same data structure,
            i.e. column names and data types same as ``data``.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        key = self._arg('key', key, str)
        index = data.index
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        self.categorical_variable = self._arg('categorical_variable',
                                              categorical_variable, ListOfStrings)
        self.strategy_by_col = self._arg('strategy_by_col',
                                         strategy_by_col, ListOfTuples)
        if self.strategy_by_col is not None:
            for col_strategy in self.strategy_by_col:
                if col_strategy[0] not in data.columns:
                    msg = ('{} is not a column name'.format(col_strategy[0]) +
                           ' of the input dataframe.')
                    logger.error(msg)
                    raise ValueError(msg)

        param_rows = [('IMPUTATION_TYPE', self.strategy, None, None),
                      ('ALS_FACTOR_NUMBER', self.als_factors, None, None),
                      ('ALS_REGULARIZATION', None, self.als_lambda, None),
                      ('ALS_MAX_ITERATION', self.als_maxit, None, None),
                      ('ALS_SEED', self.als_randomstate, None, None),
                      ('ALS_EXIT_THRESHOLD', None, self.als_exit_threshold, None),
                      ('ALS_EXIT_INTERVAL', self.als_exit_interval, None, None),
                      ('ALS_LINEAR_SYSTEM_SOLVER', self.als_linsolver, None, None),
                      ('ALS_CG_MAX_ITERATION', self.als_cg_maxit, None, None),
                      ('ALS_CENTERING', self.als_centering, None, None),
                      ('ALS_SCALING', self.als_scaling, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('HAS_ID', key is not None, None, None)]

        if self.categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, str(var))
                               for var in self.categorical_variable])
        #override the overall imputation methods for specified columns
        if self.strategy_by_col is not None:
            for col_imp_type in self.strategy_by_col:
                imp_type = self._arg('imp_type', col_imp_type[1], self.column_imputation_map)
                if len(col_imp_type) == 2:
                    param_rows.extend([('{}_IMPUTATION_TYPE'.format(col_imp_type[0]),
                                        imp_type, None, None)])
                elif len(col_imp_type) == 3:
                    if imp_type == 101:
                        param_rows.extend([('{}_IMPUTATION_TYPE'.format(col_imp_type[0]),
                                            imp_type, None, str(col_imp_type[2]))])
                    else:
                        param_rows.extend([('{}_IMPUTATION_TYPE'.format(col_imp_type[0]),
                                            imp_type, col_imp_type[2], None)])
                else:
                    continue

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'STATS_MODEL']
        outputs = ['#PAL_IMPUTATION_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, stats_model_tbl = outputs

        try:
            self._call_pal_auto(conn,
                                'PAL_MISSING_VALUE_HANDLING',
                                data,
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
        self.model_ = conn.table(stats_model_tbl)
        return conn.table(result_tbl)

    def transform(self, data, key=None, thread_ratio=None):
        """
        The function imputes missing values a DataFrame using
        statistic/model info collected from another DataFrame.

        Parameters
        ----------

        data : DataFrame
           Input DataFrame.
        key : str, optional
           Name of ID column.

           If ``key`` is not provided, then:

                - if ``data`` is indexed by a single column, then ``key`` defaults
                  to that index column;
                - otherwise, it is assumed that ``data`` contains no ID column.

        thread_ratio : float, optional
           Controls the proportion of available threads to use.

           The value range is from 0 to 1, where 0 indicates a single thread,
           and 1 indicates up to all available threads.

           Values between 0 and 1 will use up to that percentage of available threads.

           Values outside this range tell HANA PAL to heuristically determine the number of
           threads to use.

           Defaults to 0.0.

        Returns
        -------

        DataFrame
            Inputation result, structured same as ``data``.

            Statistics for the imputation result, structured as:

                - STAT_NAME: type NVACHAR(256), statistics name.
                - STAT_VALUE: type NVACHAR(5000), statistics value.

        """
        conn = data.connection_context
        require_pal_usable(conn)
        if self.model_ is None:
            raise FitIncompleteError("Stats/model not initialized. "+
                                     "Perform a fit_transform first.")
        key = self._arg('key', key, str)
        index = data.index
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)

        param_rows = [('HAS_ID', key is not None, None, None),
                      ('THREAD_RATIO', None, thread_ratio, None)]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'STATS']
        outputs = ['#PAL_IMPUTE_PREDICT_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, stats_tbl = outputs

        try:
            self._call_pal_auto(conn,
                                'PAL_MISSING_VALUE_HANDLING_WITH_MODEL',
                                data,
                                self.model_,
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
        return conn.table(result_tbl), conn.table(stats_tbl)

class Discretize(PALBase):
    """
    It is an enhanced version of binning function which can be applied to table with multiple columns.
    This function partitions table rows into multiple segments called bins, then applies smoothing
    methods in each bin of each column respectively.

    Parameters
    ----------

    strategy : {'uniform_number', 'uniform_size', 'quantile', 'sd'}
        Binning methods:
            - 'uniform_number': equal widths based on the number of bins.
            - 'uniform_size': equal widths based on the bin width.
            - 'quantile': equal number of records per bin.
            - 'sd': mean/ standard deviation bin boundaries.
    n_bins : int, optional
        Number of needed bins.

        Required and only valid when ``strategy`` is set as 'uniform_number' or 'quantile'.

        Default to 2.
    bin_size : float, optional
        Specifies the distance for binning.

        Required and only valid when ``strategy`` is set as 'uniform_size'.

        Default to 10.
    n_sd : int, optional
        Specifies the number of standard deviation at each side of the mean.

        For example, if ``n_sd`` equals 2, this function takes mean +/- 2 * standard deviation as
        the upper/lower bound for binning.

        Required and only valid when ``strategy`` is set as 'sd'.
    smoothing : {'no', 'bin_means', 'bin_medians', 'bin_boundaries'}, optional
        Specifies the default smoothing method for all non-categorical columns.

        Default to 'bin_means'.
    save_model : bool, optional
        Indicates whether the model is saved.

        Default to True.

    Attributes
    ----------
    result_ : DataFrame
        Discretize results, structured as follows:
          - ID: name as shown in input dataframe.
          - FEATURES : data smoothed respectively in each bins
    assign_ : DataFrame
        Assignment results, structured as follows:
          - ID: data ID, name as shown in input dataframe.
          - BIN_INDEX : bin index.
    model_ : DataFrame
        Model results, structured as follows:
          - ROW_INDEX: row index.
          - MODEL_CONTENT : model contents.
    stats_ : DataFrame
        Statistic results, structured as follows:
          - STAT_NAME:  statistic name.
          - STAT_VALUE: statistic value.

    Examples
    --------
    Original data:

    >>> df.collect()
        ID  ATT1   ATT2  ATT3 ATT4
    0    1  10.0  100.0   1.0    A
    1    2  10.1  101.0   1.0    A
    2    3  10.2  100.0   1.0    A
    3    4  10.4  103.0   1.0    A
    4    5  10.3  100.0   1.0    A
    5    6  40.0  400.0   4.0    C
    6    7  40.1  402.0   4.0    B
    7    8  40.2  400.0   4.0    B
    8    9  40.4  402.0   4.0    B
    9   10  40.3  400.0   4.0    A
    10  11  90.0  900.0   2.0    C
    11  12  90.1  903.0   1.0    B
    12  13  90.2  901.0   2.0    B
    13  14  90.4  900.0   1.0    B
    14  15  90.3  900.0   1.0    B

    Construct an Discretize instance:

    >>> bin = Discretize(method='uniform_number',
              n_bins=3, smoothing='bin_medians')

    Training the model with training data:

    >>> bin.fit(train_data, binning_variable='ATT1', col_smoothing=[('ATT2', 'bin_means')],
                categorical_variable='ATT3', key=None, features=None)

    >>> bin.assign_.collect()
        ID  BIN_INDEX
    0    1          1
    1    2          1
    2    3          1
    3    4          1
    4    5          1
    5    6          2
    6    7          2
    7    8          2
    8    9          2
    9   10          2
    10  11          3
    11  12          3
    12  13          3
    13  14          3
    14  15          3

    Apply the model to new data:

    >>> bin.predict(predict_data)

    >>> res.collect():
       ID  BIN_INDEX
    0   1          1
    1   2          1
    2   3          1
    3   4          1
    4   5          3
    5   6          3
    6   7          2
    """
    def __init__(self,#pylint: disable=too-many-arguments
                 strategy,
                 n_bins=None,
                 bin_size=None,
                 n_sd=None,
                 smoothing=None,
                 save_model=True):
        super(Discretize, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        self.method_map = {'uniform_number': 0, 'uniform_size': 1, 'quantile': 2, 'sd': 3}
        self.smoothing_method_map = {'no': 0, 'bin_means': 1, 'bin_medians': 2, 'bin_boundaries': 3}
        self.strategy = self._arg('strategy', strategy, self.method_map, required=True)
        self.n_bins = self._arg('n_bins', n_bins, int)
        self.bin_size = self._arg('bin_size', bin_size, float)
        self.n_sd = self._arg('n_sd', n_sd, int)#pylint:disable=invalid-name
        self.smoothing = self._arg('smoothing', smoothing, self.smoothing_method_map)#pylint:disable=line-too-long
        self.save_model = self._arg('save_model', save_model, bool)
        self.key = None
        self.features = None

    def fit(self, data, binning_variable, key=None, features=None, col_smoothing=None,#pylint:disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
            categorical_variable=None):
        """
        Fitting a Discretize model.

        Parameters
        ----------
        data : DataFrame
            Dataframe that contains the training data.
        key : str, optional
            Name of the ID column in ``data``.

            If ``key`` is not provided, then:

                - if ``data`` is indexed by a single column, then ``key`` defaults
                  to that index column;
                - otherwise, it defaults to the first column of ``data``.

        features : str/ListofStrings, optional
            Name of the feature columns which needs to be considered in the model.

            If not specified, all columns except the key column will be count as feature columns.
        binning_variable : str
            Attribute name, to which binning operation is applied.

            Variable data type must be numeric.
        col_smoothing : ListofTuples, optional
            Specifies column name and its method for smoothing, which overwrites the default smoothing method.

            For example: smoothing_method = [('ATT1', 'bin_means'), ('ATT2', 'bin_boundaries')]

            Only applies for none-categorical attributes.

            No default value.
        categorical_variable : str/ListofStrings, optional
            Indicates whether a column data is actually corresponding to a category variable even the
            data type of this column is int.

            No default value.

        Returns
        -------

        Fitted object.
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        conn = data.connection_context
        require_pal_usable(conn)
        self.key = self._arg('key', key, str)#pylint:disable=attribute-defined-outside-init
        index = data.index
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
            self.key = index if key is None else key
        cols = data.columns
        if self.key is None:
            self.key = cols[0]# pylint: disable=attribute-defined-outside-init
        cols.remove(self.key)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                self.features = self._arg('features', features, ListOfStrings)#pylint: disable=undefined-variable,attribute-defined-outside-init
            except:
                msg = ("`features` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            self.features = cols#pylint: disable=attribute-defined-outside-init
        if categorical_variable is not None:
            if isinstance(categorical_variable, str):
                categorical_variable = [categorical_variable]
            try:
                self.categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)#pylint: disable=undefined-variable,attribute-defined-outside-init
            except:
                msg = ("`categorical_variable` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        if isinstance(binning_variable, str):
            binning_variable = [binning_variable]
        try:
            self.binning_variable = self._arg('binning_variable', binning_variable, ListOfStrings)#pylint: disable=undefined-variable,attribute-defined-outside-init
        except:
            msg = ("`binning_variable` must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)
        if categorical_variable is not None:
            if not all((data.dtypes([var])[0][1] in ('INT', 'DOUBLE') and var not in categorical_variable) for var in binning_variable):
                msg = ("`binning_variable` must indicate attributes of numerical type.")
                logger.error(msg)
                raise TypeError(msg)
        self.col_smoothing = self._arg('col_smoothing', col_smoothing, ListOfTuples)# pylint: disable=attribute-defined-outside-init
        if self.col_smoothing is not None:
            for x in self.col_smoothing:#pylint:disable=invalid-name
                if len(x) != 2:#pylint:disable=bad-option-value
                    msg = ("Each tuple that specifies the smoothing method of an attribute"+
                           " should contain exactly 2 elements: 1st is attribute column name,"+
                           " 2nd is a smoothing_method code.")
                    logger.error(msg)
                    raise ValueError(msg)
                if x[1] not in self.smoothing_method_map:
                    msg = ("'{}' is not a valid smoothing method.".format(x[1]))
                    logger.error(msg)
                    raise ValueError(msg)
                if x[0] not in self.features or data.dtypes([x[0]])[0][1] not in ('INT', 'DOUBLE') or (x[0] in self.categorical_variable):#pylint:disable=line-too-long
                    msg = ("`col_smoothing` can only be applied for numerical attributes.")
                    logger.error(msg)
                    raise ValueError(msg)
        data_ = data[[self.key] + self.features]

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'ASSIGNMENT', 'MODEL', 'STATISTICS']
        tables = ['#PAL_BIN_{}_TBL_{}_{}'.format(name, self.id, unique_id) for name in tables]
        result_tbl, assign_tbl, model_tbl, statistic_tbl = tables
        param_rows = [('METHOD', self.strategy, None, None),
                      ('BIN_NUMBER', self.n_bins, None, None),
                      ('BIN_DISTANCE', None, self.bin_size, None),
                      ('SD', self.n_sd, None, None),
                      ('DEFAULT_SMOOTHING_METHOD', self.smoothing, None, None),
                      ('SAVE_MODEL', self.save_model, None, None)]
        param_rows.extend([('BINNING_VARIABLE', None, None, str(var))
                           for var in self.binning_variable])
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, str(var))
                               for var in self.categorical_variable])
        if self.col_smoothing is not None:
            param_rows.extend([('SMOOTHING_METHOD', self.smoothing_method_map[var[1]], None, var[0])
                               for var in self.col_smoothing])
        try:
            self._call_pal_auto(conn,
                                'PAL_DISCRETIZE',
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
        self.result_ = conn.table(result_tbl)# pylint: disable=attribute-defined-outside-init
        self.model_ = conn.table(model_tbl)# pylint: disable=attribute-defined-outside-init
        self.assign_ = conn.table(assign_tbl)# pylint: disable=attribute-defined-outside-init
        self.stats_ = conn.table(statistic_tbl)# pylint: disable=attribute-defined-outside-init
        return self

    def predict(self, data):
        """
        Discritizing new data using a generated Discretize model.

        Parameters
        ----------
        data : DataFrame
            Dataframe including the predict data.

        Returns
        -------
            DataFrame
                - Discretization result
                - Bin assignment
                - Statistics
        """
        conn = data.connection_context
        require_pal_usable(conn)
        if self.model_ is None:
            raise FitIncompleteError("Model not initialized.")
        data_ = data
        if (self.key is not None) and (self.features is not None):
            data_ = data[[self.key] + self.features]
            data_.index = data.index
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'ASSIGNMENT', 'STATISTICS']
        tables = ['#PAL_BIN_APPLY{}_TBL_{}_{}'.format(name, self.id, unique_id) for name in tables]
        result_tbl, assign_tbl, stats_tbl = tables
        param_rows = []
        try:
            self._call_pal_auto(conn,
                                'PAL_DISCRETIZE_APPLY',
                                data_,
                                self.model_,
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
        return conn.table(result_tbl), conn.table(assign_tbl), conn.table(stats_tbl)#pylint:disable=line-too-long

    def transform(self, data):
        """
        Data discretization using generated Discretize models.

        Parameters
        ----------
        data : DataFrame
            Dataframe including the predict data.

        Returns
        -------
            DataFrame
                - Discretization result
                - Bin assignment
                - Statistics
        """
        return self.predict(data)

    def fit_transform(self, data, binning_variable, key=None, features=None, col_smoothing=None, categorical_variable=None): #pylint: disable=too-many-arguments
        """
        Learn a discretization configuration(model) from input data and then discretize it under that configuration.

        Parameters
        ----------
        data : DataFrame
             Dataframe that contains the training data.
        key : str, optional
            Name of the ID column in ``data``.

            If ``key`` is not provided, then:

                - if ``data`` is indexed by a single column, then ``key`` defaults
                  to that index column;
                - otherwise, it is assumed that ``data`` contains no ID column.

        features : str/ListofStrings, optional
            Name of the feature columns which needs to be considered in the model.

            If not specified, all columns except the key column will be count as feature columns.
        binning_variable : str
            Attribute name, to which binning operation is applied.

            Variable data type must be numeric.
        col_smoothing : ListofTuples, optional
            Specifies column name and its method for smoothing, which overwrites the default smoothing method.

            For example: smoothing_method = [('ATT1', 'bin_means'), ('ATT2', 'bin_boundaries')]

            Only applies for non-categorical attributes.

            No default value.
        categorical_variable : str/ListofStrings, optional
            Indicates whether a column data is actually corresponding to a category variable even the
            data type of this column is int.

            No default value.

        Returns
        -------
            DataFrame
                - Discretization result
                - Bin assignment
                - Statistics
        """
        self.fit(data, binning_variable, key, features, col_smoothing, categorical_variable)
        return self.result_, self.assign_, self.stats_

class MDS(PALBase):
    r"""
    This class serves as a tool for dimensional reduction or data visualization.
    There are two kinds of input formats supported by this function: an :math:`N \times N` **dissimilarity** matrix,
    or a usual entity–feature matrix. The former is a symmetric matrix, with each element representing
    the distance (dissimilarity) between two entities, while the later can be converted to a dissimilarity matrix
    using a method specified by the user.

    Parameters
    ----------
    matrix_type : {'dissimilarity', 'observation_feature'}
        The type of the input DataFrame.

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently
        available threads.

        Values outside the range will be ignored and this function heuristically
        determines the number of threads to use.

        Default to 0.
    dim : int, optional
        The number of dimension that the input dataset is to be reduced to.

        Default to 2.
    metric : {'manhattan', 'euclidean', 'minkowski'}, optional
        The type of distance during the calculation of dissimilarity matrix.

        Only valid when ``matrix_type`` is set as 'observation_feature'.

        Default to 'euclidean'.
    minkowski_power : float, optional
        When ``metric`` is set as 'minkowski', this parameter controls the value of power.

        Only valid when ``matrix_type`` is set as 'observation_feature' and ``metric`` is set as
        'minkowski'.

        Default to 3.

    Examples
    --------
    Original data:

    >>> df.collect()
       ID        X1        X2        X3        X4
    0   1  0.000000  0.904781  0.908596  0.910306
    1   2  0.904781  0.000000  0.251446  0.597502
    2   3  0.908596  0.251446  0.000000  0.440357
    3   4  0.910306  0.597502  0.440357  0.000000

    Apply the multidimensional scaling:

    >>> mds = MDS(matrix_type='dissimilarity', dim=2, thread_ratio=0.5)
    >>> res, stats = mds.fit_transform(data=df)
    >>> res.collect()
       ID  DIMENSION     VALUE
    0   1          1  0.651917
    1   1          2 -0.015859
    2   2          1 -0.217737
    3   2          2 -0.253195
    4   3          1 -0.249907
    5   3          2 -0.072950
    6   4          1 -0.184273
    7   4          2  0.342003

    >>> stats.collect()
                              STAT_NAME  STAT_VALUE
    0                        acheived K    2.000000
    1  proportion of variation explaind    0.978901
    """
    def __init__(self, matrix_type, thread_ratio=None, dim=None, metric=None, minkowski_power=None):#pylint:disable=too-many-arguments, too-many-locals
        super(MDS, self).__init__()
        matrix_type_map = {'observation_feature': 0, 'dissimilarity': 1}
        metric_map = {'manhattan': 1, 'euclidean': 2, 'minkowski': 3}
        self.matrix_type = arg('matrix_type', matrix_type, matrix_type_map)
        self.metric = arg('metric', metric, metric_map)
        if metric is not None and matrix_type != matrix_type_map['observation_feature']:
            msg = ("`metric` is invalid when input matrix_type is not observation_feature")
            logger.error(msg)
            raise ValueError(msg)
        self.thread_ratio = arg('thread_ratio', thread_ratio, float)
        self.dim = arg('dim', dim, int)
        self.minkowski_power = arg('minkowski_power', minkowski_power, float)
        if minkowski_power is not None and metric != metric_map['minkowski']:
            msg = ("`minkowski_power` is invalid when input metric " + \
            "is not set as 'minkowski'.")
            logger.error(msg)
            raise ValueError(msg)

    def fit_transform(self, data, key=None, features=None):
        """
        Scaling of given datasets in multiple dimensions.

        Parameters
        ----------
        data : DataFrame
             Dataframe that contains the training data.
        key : str, optional
            Name of the ID column ``data``.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : str/ListofStrings, optional
            Name of the feature columns which needs to be considered in the model.

            If not specified, all columns except the key column will be count as feature columns.

        Returns
        -------
            DataFrame
                - Scaling result of `data`, structured as follows:
                    - Data ID : IDs from `data`
                    - DIMENSION : The dimension number in `data`
                    - VALUE : Scaled value
                - Statistics
        """
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        index = data.index
        key = self._arg('key', key, str, required=not isinstance(index, str))
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
        key = index if key is None else key
        cols.remove(key)
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            except:
                msg = ("`features` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        else:
            features = cols
        data_ = data[[key] + features]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['RESULT', 'STATS']
        tables = ['#MDS_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
        res_tbl, stats_tbl = tables
        param_rows = [('K', self.dim, None, None),
                      ('INPUT_TYPE', self.matrix_type, None, None),
                      ('DISTANCE_LEVEL', self.metric, None, None),
                      ('MINKOWSKI_POWER', None, self.minkowski_power, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None)]
        try:
            self._call_pal_auto(conn,
                                "PAL_MDS",
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
        return conn.table(res_tbl), conn.table(stats_tbl)

class Sampling(PALBase):#pylint:disable=too-many-arguments, too-many-locals
    """
    This class is used to choose a small portion of the records as representatives.

    Parameters
    ----------

    method : str
        Specifies the sampling method.

        Valid options include:
        'first_n', 'middle_n', 'last_n', 'every_nth', 'simple_random_with_replacement',
        'simple_random_without_replacement', 'systematic', 'stratified_with_replacement',
        'stratified_without_replacement'.

        For the random methods, the system time is used for the seed.

    interval : int, optional
        The interval between two samples.

        Only required when ``method`` is 'every_nth'.

        If this parameter is not specified, the ``sampling_size`` parameter will be used.

    sampling_size : int, optional
        Number of the samples.

        Default to 1.

    random_state : int, optional
        Indicates the seed used to initialize the random number generator.

        It can be set to 0 or a positive value, where:
            - 0: Uses the system time
            - Others: Uses the specified seed

        Default to 0.

    percentage : float, optional
        Percentage of the samples.

        Use this parameter when sampling_size is not set.

        If both ``sampling_size`` and ``percentage`` are specified, ``percentage`` takes precedence.

        Default to 0.1.

    Examples
    --------
    Original data:

    >>> df.collect().head(10)
        EMPNO  GENDER  INCOME
    0       1    male  4000.5
    1       2    male  5000.7
    2       3  female  5100.8
    3       4    male  5400.9
    4       5  female  5500.2
    5       6    male  5540.4
    6       7    male  4500.9
    7       8  female  6000.8
    8       9    male  7120.8
    9      10  female  8120.9

    Apply the sampling function:

    >>> smp = Sampling(method='every_nth', interval=5, sampling_size=8)
    >>> res = smp.fit_transform(data=df)
    >>> res.collect()
       EMPNO  GENDER  INCOME
    0      5  female  5500.2
    1     10  female  8120.9
    2     15    male  9876.5
    3     20  female  8705.7
    4     25  female  8794.9

    """
    def __init__(self, method, interval=None, sampling_size=None, random_state=None, percentage=None):#pylint:disable=too-many-arguments, too-many-locals
        super(Sampling, self).__init__()
        method_map = {'first_n': 0, 'middle_n': 1, 'last_n': 2, 'every_nth': 3,
                      'simple_random_with_replacement': 4, 'simple_random_without_replacement': 5,
                      'systematic': 6, 'stratified_with_replacement': 7,
                      'stratified_without_replacement': 8}
        self.method = arg('method', method, method_map)
        self.interval = arg('interval', interval, int)
        self.sampling_size = arg('sampling_size', sampling_size, int)
        self.random_state = arg('random_state', random_state, int)
        self.percentage = arg('percentage', percentage, float)
        if method == 3 and interval is None:
            msg = ("`interval` is required when `method` is set as 'every_nth'.")
            logger.error(msg)
            raise ValueError(msg)
        if percentage is not None and sampling_size is not None:
            sampling_size = None

    def fit_transform(self, data, features=None):
        """
        Samping the input dataset under specified configuration.

        Parameters
        ----------
        data : DataFrame
            Input Dataframe.

        features : str/ListofStrings, optional
            The column that is used to do the stratified sampling.

            Only required when method is 'stratified_with_replacement',
            or 'stratified_without_replacement'.

            Defaults to None.

        Returns
        -------
        DataFrame
            Sampling results, same structure as defined in the Input DataFrame.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        column_choose = None
        if features is not None:
            if isinstance(features, str):
                features = [features]
            try:
                features = arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
                column_choose = []
                for feature in features:
                    column_choose.append(data.columns.index(feature))
            except:
                msg = ("`features` must be list of string or string.")
                logger.error(msg)
                raise TypeError(msg)
        if self.method in (7, 8) and column_choose is None:
            msg = ("`features` specification is required when `method` " + \
                "is set to 'stratified_with_replacement' or 'stratified_without_replacement'.")
            logger.error(msg)
            raise ValueError(msg)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        res_tbl = "#SAMPLING_RESULT_TBL_{}".format(unique_id)
        param_rows = [('SAMPLING_METHOD', self.method, None, None),
                      ('INTERVAL', self.interval, None, None),
                      ('SAMPLING_SIZE', self.sampling_size, None, None),
                      ('RANDOM_SEED', self.random_state, None, None),
                      ('PERCENTAGE', None, self.percentage, None)]
        if column_choose is not None:
            param_rows.extend([('COLUMN_CHOOSE', col+1, None, None)
                               for col in column_choose])
        try:
            self._call_pal_auto(conn,
                                "PAL_SAMPLING",
                                data,
                                ParameterTable().with_data(param_rows),
                                res_tbl)
        except dbapi.Error as db_err:
            try_drop(conn, res_tbl)
            logger.exception(str(db_err))
            raise
        except pyodbc.Error as db_err:
            try_drop(conn, res_tbl)
            logger.exception(str(db_err.args[1]))
            raise
        return conn.table(res_tbl)

class SMOTE(PALBase): #pylint:disable=too-many-arguments, too-many-locals
    """
    This class is to handle imbalanced dataset. Synthetic minority over-sampling technique (SMOTE)
    proposes an over-sampling approach in which the minority class is over-sampled by creating
    "synthetic" examples in "feature space".

    Parameters
    ----------

    smote_amount : int, optional
        Amount of SMOTE N%. E.g. 200 means 200%, so each minority class sample will generate 2 synthetic samples.

        The synthetic samples are generated until the minority class sample amount matches
        the majority class sample amount.

    k_nearest_neighbours : int, optional
        Number of nearest neighbors (k).

        Defaults to 1.
    minority_class : str, optional(deprecated)
        Specifies the minority class value in dependent variable column.

        All classes except majority class are re-sampled to match the majority class sample amount.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently
        available threads.

        Values outside the range [0, 1] will be ignored and this function heuristically determines the number of threads to use.

        Default to 0.
    random_seed : int, optional
        Specifies the seed for random number generator.

          - 0: Uses the current time (in seconds) as seed
          - Others: Uses the specified value as seed

        Defaults to 0.
    method : int, optional(deprecated)
        Searching method when finding K nearest neighbour.

          - 0: Brute force searching
          - 1: KD-tree searching

        Defaults to 0.
    search_method : str, optional
        Specifies the searching method for finding the k nearest-neighbors.

          - 'brute-force'
          - 'kd-tree'

        Defaults to 'brute-force'.

    category_weights : float, optional
        Represents the weight of category attributes.
        The value must be greater or equal to 0.

    Examples
    --------
    >>> smote = SMOTE(smote_amount=200, k_nearest_neighbours=2,
                      search_method='kd-tree')
    >>> res = smote.fit_transform(data=df, label = 'TYPE', minority_class=2)
    """
    search_method_map = {'brute-force' : 0, 'kd-tree' : 1}

    def __init__(self, smote_amount=None,#pylint:disable=too-many-arguments, too-many-locals
                 k_nearest_neighbours=None,
                 minority_class=None,
                 thread_ratio=None,
                 random_seed=None,
                 method=None,
                 search_method=None,
                 category_weights=None):
        super(SMOTE, self).__init__()
        self.smote_amount = self._arg('smote_amount', smote_amount, int)
        self.k_nearest_neighbours = self._arg('k_nearest_neighbours', k_nearest_neighbours,
                                              int)
        self.minority_class = self._arg('minority_class', minority_class, (str, int))
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.random_seed = self._arg('random_seed', random_seed, int)
        self.method = self._arg('method', method, int)
        self.search_method = self._arg('search_method', search_method,
                                       self.search_method_map)
        self.category_weights = self._arg('category_weights', category_weights,
                                          (float, int))
        #self.label = None

    def fit_transform(self, data, label,#pylint:disable=too-many-arguments
                      minority_class=None,
                      categorical_variable=None,
                      variable_weight=None):
        """
        Upsamping given datasets using SMOTE with specified configuration.

        Parameters
        ----------
        data : DataFrame
             Dataframe that contains the training data.
        label : str
            Specifies the dependent variable by name.
        minority_class : str/int, optional
            Specifies the minority class value in dependent variable column.

            If not specified, all but the majority classes are resampled to match the majority class sample amount.

        categorical_variable : str/ListOfStrings, optional
            Specifies the list of INTEGER columns that should be treated as categorical.

            By default, only VARCHAR and NVARCHAR columns are treated as categorical,
            while numerical (i.e. INTEGER or DOUBLE) columns are treated as continuous.

            No default value.

        variable_weight : dict, optional
            Specifies the weights of variables participating in distance calculation in a dictionary,
            illustrated as follows:

            {variable_name0 : value0, variable_name1 : value1, ...}.

            The values must be no less than 0.

            Weights default to 1 for variables not specified.

        Returns
        -------
        DataFrame
             - SMOTE result, the same structure as defined in the input data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        label = arg('label', label, str)
        label_type = data.dtypes([label])
        if "INT" in label_type[0][1]:
            minority_class = arg('minority_class', minority_class, int)
        else:
            minority_class = arg('minority_class', minority_class, str)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable,
                                         ListOfStrings)
        variable_weight = self._arg('variable_weight', variable_weight, dict)
        tables = ['RESULT']
        tables = ['#SMOTE_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
        res_tbl = tables[0]

        param_rows = [('SMOTE_AMOUNT', self.smote_amount, None, None),
                      ('K_NEAREST_NEIGHBOURS', self.k_nearest_neighbours, None, None),
                      ('DEPENDENT_VARIABLE', None, None, label),
                      ('MINORITY_CLASS', None, None,
                       self.minority_class if minority_class is None else minority_class),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('RANDOM_SEED', self.random_seed, None, None),
                      ('METHOD',
                       self.method if self.search_method is None else self.search_method,
                       None, None),
                      ('CATEGORY_WEIGHTS', None, None, self.category_weights)]
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None,
                                None, var) for var in categorical_variable])
        if variable_weight is not None:
            param_rows.extend([('VARIABLE_WEIGHT', None,
                                variable_weight[var], var) for var in variable_weight])
        try:
            self._call_pal_auto(conn,
                                "PAL_SMOTE",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        return conn.table(res_tbl)

class SMOTETomek(PALBase): #pylint:disable=too-many-arguments, too-many-locals
    """
    This class combines over-sampling using SMOTE and cleaning(under-sampling) using Tomek links.

    Parameters
    ----------

    smote_amount : int, optional
        Amount of SMOTE N%. E.g. 200 means 200%, so each minority class sample will generate 2 synthetic samples.

        The synthetic samples are generated until the minority class sample amount matches
        the majority class sample amount.
    k_nearest_neighbours : int, optional
        Number of nearest neighbors (k).

        Defaults to 1.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently
        available threads.

        Values outside the range [0, 1] will be ignored and this function heuristically determines the number of threads to use.

        Default to 0.
    random_seed : int, optional
        Specifies the seed for random number generator.

          - 0: Uses the current time (in second) as seed
          - Others: Uses the specified value as seed

        Defaults to 0.
    search_method : str, optional
        Specifies the searching method when finding K nearest neighbour.

          - 'brute-force'
          - 'kd-tree'

        Defaults to 'brute-force'.
    sampling_strategy : str, optional
        Specifies the classes targeted by resampling:

          - 'majority' : resamples only the majority class
          - 'non-minority' : resamples all classes except the minority class
          - 'non-majority' : resamples all classes except the majority class
          - 'all' : resamples all classes

        Defaults to 'majority'.

    category_weights : float, optional
        Represents the weight of category attributes.
        The value must be greater or equal to 0.

    Examples
    --------
    >>> smotetomek = SMOTETomek(smote_amount=200,
                                k_nearest_neighbours=2,
                                random_seed=2,
                                search_method='kd-tree',
                                sampling_strategy='all')
    >>> res = smotetomek.fit_transform(data=df, label='TYPE', minority_class=2)
    """
    method_map = {'brute-force': 0, 'kd-tree' : 1}
    strategy_map = {'majority': 0, 'non-minority' : 1, 'non-majority' : 2, 'all' : 3}

    def __init__(self, smote_amount=None,#pylint:disable=too-many-arguments, too-many-locals
                 k_nearest_neighbours=None,
                 thread_ratio=None,
                 random_seed=None,
                 search_method=None,
                 sampling_strategy=None,
                 category_weights=None):
        super(SMOTETomek, self).__init__()
        self.smote_amount = self._arg('smote_amount', smote_amount, int)
        self.k_nearest_neighbours = self._arg('k_nearest_neighbours', k_nearest_neighbours, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.random_seed = self._arg('random_seed', random_seed, int)
        self.method = self._arg('search_method', search_method, self.method_map)
        self.sampling_strategy = self._arg('sampling_strategy', sampling_strategy, self.strategy_map)
        self.category_weights = self._arg('category_weights', category_weights, (float, int))

    def fit_transform(self, data, label,#pylint:disable=too-many-arguments
                      minority_class=None,
                      categorical_variable=None,
                      variable_weight=None):
        """
        Perform both over-sampling using SMOTE and under-sampling by removing Tomek's links on given datasets.

        Parameters
        ----------
        data : DataFrame
             Dataframe that contains the training data.
        label : str
            Specifies the dependent variable by name.
        minority_class : str/int, optional
            Specifies the minority class value in dependent variable column.

            If not specified, all but the majority classes are resampled to match the majority class sample amount.

        categorical_variable : str/ListOfStrings, optional
            Specifies the list of INTEGER columns that should be treated as categorical.

            By default, only VARCHAR and NVARCHAR columns are treated as categorical,
            while numerical (i.e. INTEGER or DOUBLE) columns are treated as continuous.

            No default value.

        variable_weight : dict, optional
            Specifies the weights of variables participating in distance calculation in a dictionary:

              - key : variable(column) name
              - value : weight for distance calculation

            No default value.

        Returns
        -------
        DataFrame
            SMOTETomek result, the same structure as defined in the input data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        label = self._arg('label', label, str)
        label_type = data.dtypes([label])
        if "INT" in label_type[0][1]:
            minority_class = self._arg('minority_class', minority_class, int)
        else:
            minority_class = self._arg('minority_class', minority_class, str)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable,
                                         ListOfStrings)
        variable_weight = self._arg('variable_weight', variable_weight, dict)
        tables = ['RESULT']
        tables = ['#SMOTETOMEK_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
        res_tbl = tables[0]
        param_rows = [('SMOTE_AMOUNT', self.smote_amount, None, None),
                      ('K_NEAREST_NEIGHBOURS', self.k_nearest_neighbours, None, None),
                      ('DEPENDENT_VARIABLE', None, None, label),
                      ('MINORITY_CLASS', None, None, str(minority_class)),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('RANDOM_SEED', self.random_seed, None, None),
                      ('METHOD', self.method, None, None),
                      ('SAMPLING_STRATEGY', self.sampling_strategy, None, None),
                      ('CATEGORY_WEIGHTS', None, self.category_weights, None)]
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None,
                                None, var) for var in categorical_variable])
        if variable_weight is not None:
            param_rows.extend([('VARIABLE_WEIGHT', None,
                                variable_weight[var], var) for var in variable_weight])
        try:
            self._call_pal_auto(conn,
                                "PAL_SMOTETOMEK",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        return conn.table(res_tbl)

class TomekLinks(PALBase): #pylint:disable=too-many-arguments, too-many-locals
    """
    This class is for performing under-sampling by removing Tomek's links.

    Parameters
    ----------

    distance_level : str, optional
        Specifies the distance method between train data and test data point.

          - 'manhattan'
          - 'euclidean'
          - 'minkowski'
          - 'chebyshev'
          - 'consine'

        Defaults to 'euclidean'.
    minkowski_power : float, optional
        Specifies the value of power for Minkowski distance calculation.

        Defaults to 3.

        Valid only when ``distance_level`` is 'minkowski'.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently
        available threads.

        Values outside the range [0, 1] will be ignored and this function heuristically
        determines the number of threads to use.

        Default to 0.
    search_method : str, optional
        Specifies the searching method when finding K nearest neighbour.

          - 'brute-force'
          - 'kd-tree'

        Defaults to 'brute-force'.
    sampling_strategy : str, optional

        Specifies the classes targeted by resampling:

          - 'majority' : resamples only the minority class
          - 'non-minority' : resamples all classes except the minority class
          - 'non-majority' : resamples all classes except the majority class
          - 'all' : resamples all classes

        Defaults to 'majority'

    category_weights : float, optional
        Specifies the weight for categorical attributes.

        Defaults to 0.707 if not provided.

    Examples
    --------
    >>> tomeklinks = TomekLinks(search_method='kd-tree',
                                sampling_strategy='majority')
    >>> res = smotetomek.fit_transform(data=df, label='TYPE')
    """
    method_map = {'brute-force': 0, 'kd-tree' : 1}
    strategy_map = {'minority': 0, 'non-minority' : 1, 'non-majority' : 2, 'all' : 3}
    distance_map = {'manhattan' : 1, 'euclidean' : 2, 'minkowski' : 3, 'chebyshev' : 4, 'cosine' : 6}

    def __init__(self, distance_level=None,#pylint:disable=too-many-arguments, too-many-locals
                 minkowski_power=None,
                 thread_ratio=None,
                 search_method=None,
                 sampling_strategy=None,
                 category_weights=None):
        super(TomekLinks, self).__init__()
        self.distance_level = self._arg('distance_level', distance_level, self.distance_map)
        self.minkowski_power = self._arg('minkowski_power', minkowski_power, float)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.search_method = self._arg('search_method', search_method, self.method_map)
        self.sampling_strategy = self._arg('sampling_strategy', sampling_strategy, self.strategy_map)
        self.category_weights = self._arg('category_weights', category_weights, float)

    def fit_transform(self, data,#pylint:disable=too-many-arguments, too-many-locals
                      key=None,
                      label=None,
                      categorical_variable=None,
                      variable_weight=None):
        """
        Perform under-sampling on given datasets by removing Tomek's links.

        Parameters
        ----------
        data : DataFrame
            Dataframe that contains the training data.
        key : str, optional
            Specifies the name of the ID column.

            If ``key`` is not provided, then:

                - if ``data`` is indexed by a single column, then ``key`` defaults
                  to that index column;
                - otherwise, it is assumed that ``data`` contains no ID column.

        label : str
            Specifies the dependent variable by name.
        categorical_variable : str/ListOfStrings, optional
            Specifies the list of INTEGER columns that should be treated as categorical.

            By default, only VARCHAR and NVARCHAR columns are treated as categorical,
            while numerical (i.e. INTEGER or DOUBLE) columns are treated as continuous.

            No default value.

        variable_weight : dict, optional
            Specifies the weights of variables participating in distance calculation in a dictionary:

              - key : variable(column) name
              - value : weight for distance calculation

            No default value.

        Returns
        -------
        DataFrame
            - Undersampled result, the same structure as defined in the input data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        key = self._arg('key', key, str)
        index = data.index
        if isinstance(index, str):
            if key is not None and index != key:
                msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                "and the designated index column '{}'.".format(index)
                logger.warning(msg)
            key = index if key is None else key
        label = self._arg('label', label, str, required=True)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable',
                                         categorical_variable,
                                         ListOfStrings)
        orig_cols = data.columns
        if key is not None:
            cols = orig_cols
            cols.remove(key)
            data = data[[key] + cols]
        tables = ['RESULT']
        tables = ['#TOMEK_LINKS_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
        res_tbl = tables[0]
        param_rows = [('DISTANCE_LEVEL', self.distance_level, None, None),
                      ('HAS_ID', key is not None, None, None),
                      ('MINKOWSKI_POWER', None, self.minkowski_power, None),
                      ('DEPENDENT_VARIABLE', None, None, label),
                      ('CATEGORY_WEIGHTS', None, self.category_weights, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('METHOD', self.search_method, None, None),
                      ('SAMPLING_STRATEGY', self.sampling_strategy, None, None)]
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE',
                                None, None, var) for var in categorical_variable])
        if variable_weight is not None:
            param_rows.extend([('VARIABLE_WEIGHT',
                                None,
                                variable_weight[varb],
                                varb) for varb in variable_weight])
        try:
            self._call_pal_auto(conn,
                                "PAL_TOMEKLINKS",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        return conn.table(res_tbl)[orig_cols]#consistency between input & output

@deprecated(version='1.0.8', reason="This method is deprecated. Please use MDS instead.")
def mds(data, matrix_type, thread_ratio=None,#pylint:disable=too-many-arguments, too-many-locals
        dim=None, metric=None, minkowski_power=None,
        key=None, features=None):
    """
    This function serves as a tool for dimensional reduction or data visualization.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.
    matrix_type : {'dissimilarity', 'observation_feature'}
        The type of the input table.
        Mandatory.
    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread, and 1 means using at most all the currently
        available threads.

        Values outside the range will be ignored and this function heuristically determines the number of threads to use.

        Default to 0.
    dim : int, optional
        The number of dimension that the input dataset is to be reduced to.

        Default to 2.
    metric : {'manhattan', 'euclidean', 'minkowski'}, optional
        The type of distance during the calculation of dissimilarity matrix.

        Only valid when ``matrix_type`` is set as 'observation_feature'.

        Default to 'euclidean'.
    minkowski_power : float, optional
        When ``metric`` is  'minkowski', this parameter controls the value of power.

        Only valid when ``matrix_type`` is set as 'observation_feature' and ``metric`` is set as
        'minkowski'.

        Default to 3.
    key : str, optional
        Name of the ID column in the dataframe.

        If not specified, the first col will be taken as the ID column.
    features : str/ListOfStrings, optional
        Name of the feature column in the dataframe.

        If not specified, columns except the ID column will be taken as feature columns.

    Returns
    -------
    DataFrame
        - Sampling results, structured as follows:
            - DATA_ID: name as shown in input dataframe.
            - DIMENSION: dimension.
            - VALUE: value.
        - Statistic results, structured as follows:
            - STAT_NAME:  statistic name.
            - STAT_VALUE: statistic value.

    Examples
    --------
    Original data:

    >>> df.collect()
         ID        X1        X2        X3        X4
        0   1  0.000000  0.904781  0.908596  0.910306
        1   2  0.904781  0.000000  0.251446  0.597502
        2   3  0.908596  0.251446  0.000000  0.440357
        3   4  0.910306  0.597502  0.440357  0.000000

    Apply the multidimensional scaling:

    >>> res,stats = mds(data=df,
                        matrix_type='dissimilarity', dim=2, thread_ratio=0.5)
    >>> res.collect()
               ID  DIMENSION     VALUE
        0   1          1  0.651917
        1   1          2 -0.015859
        2   2          1 -0.217737
        3   2          2 -0.253195
        4   3          1 -0.249907
        5   3          2 -0.072950
        6   4          1 -0.184273
        7   4          2  0.342003

    >>> stats.collect()
                                  STAT_NAME  STAT_VALUE
        0                        acheived K    2.000000
        1  proportion of variation explaind    0.978901
    """
    conn = data.connection_context
    require_pal_usable(conn)
    matrix_type_map = {'observation_feature': 0, 'dissimilarity': 1}
    metric_map = {'manhattan': 1, 'euclidean': 2, 'minkowski': 3}

    matrix_type = arg('matrix_type', matrix_type, matrix_type_map)
    metric = arg('metric', metric, metric_map)
    if metric is not None and matrix_type != matrix_type_map['observation_feature']:
        msg = ("`metric` is invalid when input matrix_type is not observation_feature")
        logger.error(msg)
        raise ValueError(msg)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    dim = arg('dim', dim, int)
    minkowski_power = arg('minkowski_power', minkowski_power, float)
    if minkowski_power is not None and metric != metric_map['minkowski']:
        msg = ("`minkowski_power` is invalid when input metric "+
               "is not set as 'minkowski'.")
        logger.error(msg)
        raise ValueError(msg)
    cols = data.columns
    key = arg('key', key, str)
    if key is None:
        key = cols[0]
    cols.remove(key)
    if features is not None:
        if isinstance(features, str):
            features = [features]
        try:
            features = arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
        except:
            msg = ("`features` must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)
    else:
        features = cols
    data_ = data[[key] + features]

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    tables = ['RESULT', 'STATS']
    tables = ['#MDS_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
    res_tbl, stats_tbl = tables
    param_rows = [('K', dim, None, None),
                  ('INPUT_TYPE', matrix_type, None, None),
                  ('DISTANCE_LEVEL', metric, None, None),
                  ('MINKOWSKI_POWER', None, minkowski_power, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]
    try:
        call_pal_auto_with_hint(conn,
                                None,
                                "PAL_MDS",
                                data_,
                                ParameterTable().with_data(param_rows),
                                *tables)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, tables)
        raise
    return conn.table(res_tbl), conn.table(stats_tbl)

@deprecated(version='1.0.8', reason="This method is deprecated. Please use Sampling instead.")
def sampling(data, method, interval=None, features=None, sampling_size=None, random_state=None, percentage=None): #pylint:disable=too-many-arguments, too-many-locals
    """
    This function is used to choose a small portion of the records as representatives.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.
    method : str
        Specifies the sampling method.
        Valid options include:

        'first_n', 'middle_n', 'last_n', 'every_nth', 'simple_random_with_replacement',
        'simple_random_without_replacement', 'systematic', 'stratified_with_replacement',
        'stratified_without_replacement'.

        For the random methods, the system time is used for the seed.
    interval : int, optional
        The interval between two samples.

        Only required when ``method`` is 'every_nth'.

        If this parameter is not specified, the sampling_size parameter will be used.
    features : str/ListofStrings, optional
        The column that is used to do the stratified sampling.

        Only required when method is stratified_with_replacement,
        or stratified_without_replacement.
    sampling_size : int, optional
        Number of the samples.

        Default to 1.
    random_state : int, optional
        Indicates the seed used to initialize the random number generator.

        It can be set to 0 or a positive value.

            - 0: Uses the system time
            - Not 0: Uses the specified seed

        Default to 0.
    percentage : float, optional
        Percentage of the samples.

        Use this parameter when sampling_size is not set.

        If both sampling_size and percentage are specified, percentage takes precedence.

        Default to 0.1.

    Returns
    -------
    DataFrame
        - Sampling results, structured as follows:
            - DATA_FEATURES: same structure as defined in the Input Table.

    Examples
    --------
    Original data:

    >>> df.collect().head(10)
             EMPNO  GENDER  INCOME
        0       1    male  4000.5
        1       2    male  5000.7
        2       3  female  5100.8
        3       4    male  5400.9
        4       5  female  5500.2
        5       6    male  5540.4
        6       7    male  4500.9
        7       8  female  6000.8
        8       9    male  7120.8
        9      10  female  8120.9

    Apply the sampling function:

    >>> res = sampling(data=df, method='every_nth', interval=5, sampling_size=8)

    >>> res.collect()
             EMPNO  GENDER  INCOME
        0      5  female  5500.2
        1     10  female  8120.9
        2     15    male  9876.5
        3     20  female  8705.7
        4     25  female  8794.9
    """
    conn = data.connection_context
    require_pal_usable(conn)
    method_map = {'first_n': 0, 'middle_n': 1, 'last_n': 2, 'every_nth': 3,
                  'simple_random_with_replacement': 4, 'simple_random_without_replacement': 5,
                  'systematic': 6, 'stratified_with_replacement': 7,
                  'stratified_without_replacement': 8}
    method = arg('method', method, method_map)
    interval = arg('interval', interval, int)
    column_choose = None
    if features is not None:
        if isinstance(features, str):
            features = [features]
        try:
            features = arg('features', features, ListOfStrings)#pylint: disable=undefined-variable
            column_choose = []
            for feature in features:
                column_choose.append(data.columns.index(feature))
        except:
            msg = ("`features` must be list of string or string.")
            logger.error(msg)
            raise TypeError(msg)
    sampling_size = arg('sampling_size', sampling_size, int)
    random_state = arg('random_state', random_state, int)
    percentage = arg('percentage', percentage, float)

    if method == 3 and interval is None:
        msg = ("`interval` is required when `method` is set as 'every_nth'.")
        logger.error(msg)
        raise ValueError(msg)
    if method in (7, 8) and column_choose is None:
        msg = ("`features` specification is required when `method` "+
               "is set to 'stratified_with_replacement' or 'stratified_without_replacement'.")
        logger.error(msg)
        raise ValueError(msg)
    if percentage is not None and sampling_size is not None:
        sampling_size = None

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()

    res_tbl = "#SAMPLING_RESULT_TBL_{}".format(unique_id)
    param_rows = [('SAMPLING_METHOD', method, None, None),
                  ('INTERVAL', interval, None, None),
                  ('SAMPLING_SIZE', sampling_size, None, None),
                  ('RANDOM_SEED', random_state, None, None),
                  ('PERCENTAGE', None, percentage, None)]
    if column_choose is not None:
        param_rows.extend([('COLUMN_CHOOSE', col+1, None, None)
                           for col in column_choose])
    try:
        call_pal_auto_with_hint(conn,
                                None,
                                "PAL_SAMPLING",
                                data,
                                ParameterTable().with_data(param_rows),
                                res_tbl)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        raise
    except pyodbc.Error as db_err:
        logger.exception(str(db_err.args[1]))
        raise
    return conn.table(res_tbl)

def variance_test(data, sigma_num, thread_ratio=None, key=None, data_col=None):#pylint:disable=too-many-arguments
    """
    Variance Test is a method to identify the outliers of n number of numeric data {xi} where 0 < i < n+1,
    using the mean and the standard deviation of n number of numeric data.

    Parameters
    ----------

    data : DataFrame
        DataFrame containing the data.

    sigama_num : float
        Multiplier for sigma.

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.

        The value range is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.

        Values outside the range will be ignored and this function
        heuristically determines the number of threads to use.

        Default to 0.
    key : str, optional
        Name of the ID column in ``data``.

        If ``key`` is not specified, then:

            - if ``data`` is indexed by a single column, then ``key`` defaults
              to that index column;
            - otherwise, it defaults to the first column of ``data``.

    data_col : str, optional
        Name of the raw data column in the dataframe.

        If not specified, defaults to the last column of data.

    Returns
    -------
    DataFrame
        - Sampling results, structured as follows:
            - DATA_ID: name as shown in input dataframe.
            - IS_OUT_OF_RANGE: 0 -> in bounds, 1 -> out of bounds.
        - Statistic results, structured as follows:
            - STAT_NAME:  statistic name.
            - STAT_VALUE: statistic value.

    Examples
    --------
    Original data:

    >>> df.collect().tail(10)
            ID      X
        10  10   26.0
        11  11   28.0
        12  12   29.0
        13  13   27.0
        14  14   26.0
        15  15   23.0
        16  16   22.0
        17  17   23.0
        18  18   25.0
        19  19  103.0

    Apply the variance test:

    >>> res, stats = variance_test(data, sigma_num=3.0)

    >>> res.collect().tail(10)
            ID  IS_OUT_OF_RANGE
        10  10                0
        11  11                0
        12  12                0
        13  13                0
        14  14                0
        15  15                0
        16  16                0
        17  17                0
        18  18                0
        19  19                1
    >>> stats.collect()
            STAT_NAME  STAT_VALUE
        0        mean   28.400000
    """
    conn = data.connection_context
    require_pal_usable(conn)
    sigma_num = arg('sigma_num', sigma_num, float)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    key = arg('key', key, str)
    index = data.index
    if isinstance(index, str):
        if key is not None and index != key:
            msg = "Discrepancy between the designated key column '{}' ".format(key) +\
            "and the designated index column '{}'.".format(index)
            logger.warning(msg)
        key = index if key is None else key
    if key is None:
        key = data.columns[0]
    data_col = arg('data_col', data_col, str)
    if data_col is None:
        data_col = data.columns[-1]
    if key == data_col:
        msg = ("Input data should have at least two columns, "+
               "including the ID column and the data column.")
        logger.error(msg)
        raise ValueError(msg)
    data_ = data[[key] + [data_col]]

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    tables = ['RESULT', 'STATS']
    tables = ['#VARIENCE_TEST_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
    res_tbl, stats_tbl = tables
    param_rows = [('SIGMA_NUM', None, sigma_num, None),
                  ('THREAD_RATIO', None, thread_ratio, None)]
    try:
        call_pal_auto_with_hint(conn,
                                None,
                                "PAL_VARIANCE_TEST",
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
    return conn.table(res_tbl), conn.table(stats_tbl)

class FeatureSelection(PALBase):
    r"""
    Feature selection(FS) is a dimensionality reduction technique, which selects a subset of revevant features for model construction,
    thus reducing the memory storage and imporving computational efficiency while avoiding significant loss of infromation.

    Parameters
    ----------

    fs_method : {'anova', 'chi-squared', 'gini-index', 'fisher-score', 'information-gain', 'MRMR', 'JMI', 'IWFS', 'FCBF', 'laplacian-score', 'SPEC', 'ReliefF', 'ADMM', 'CSO'}

        - Statistical based FS methods:
            - 'anova':Anova.
            - 'chi-squared': Chi-squared.
            - 'gini-index': Gini Index.
            - 'fisher-score': Fisher Score.
        - Information theoretical based FS methods:
            - 'information-gain': Information Gain.
            - 'MRMR': Minimum Redundancy Maximum Relevance.
            - 'JMI': Joint Mutual Infromation.
            - 'IWFS': Interaction Weight Based Feature Selection.
            - 'FCBF': Fast Correlation Based Filter.
        - Similarity based FS methods:
            - 'laplacian-score': Laplacian Score.
            - 'SPEC': Spectral Feature Selection.
            - 'ReliefF': ReliefF.
        - Sparse Learning Based FS method:
            - 'ADMM': ADMM.
        - Wrapper method:
            - 'CSO': Competitive Swarm Optimizer.
    top_k_best : int, optional
        Top k features to be selected. Must be assigned a value except for FCBF and CSO. It will not affect FCBF and CSO.
    thread_ratio, float, optional
        The ratio of available threads.
        - 0: single thread
        - 0~1: percentage
        - others: heuristically determined

        Defaults to -1.
    seed :  int, optional
        Random seed. 0 means using system time as seed.

        Defaults to 0.
    fs_threshold : float, optional
        Predefined threshold for symmetrical uncertainty(SU) values between features and target. Used in FCBF.

        Defaults to 0.01.
    fs_n_neighbours : int, optional
        Number of neighbours considered in the computation of affinity matirx. Used in similarity based FS method.

        Defaults to 5.
    fs_category_weight : float, optional
        The weight of categorical features whilst calculating distance. Used in similarity based FS method.

        Defaults to 0.5*avg(all numerical columns's std)
    fs_sigma : float, optional
        Sigma in affinity matrix. Used in similarity based FS method.

        Defaults to 1.0.
    fs_regularization_power : int, optional
        The order of the power function that penalizes high frequency components. Used in SPEC.

        Defaults to 0.
    fs_rowsampling_ratio : float, optional
        The ratio of random sampling without replacement. Used in ReliefF, ADMM and CSO.

        Defaults to 0.6 in ReliefF, 1.0 in ADMM and CSO.
    fs_max_iter : int, opitional
        Maximal iterations allowed to run optimization. Used in ADMM.

        Defaults to 100.
    fs_admm_tol : float, optional
        Convergence threshold. Used in ADMM.

        Defaults to 0.0001.
    fs_admm_rho : float, optional
        Lagrangian Multiplier. Used in ADMM.

        Defaults to 1.0.
    fs_admm_mu : float, optional
        Gain of fs_admm_rho at each iteration. Used in ADMM.

        Defaults to 1.05.
    fs_admm_gamma : float, optional
        Regularization coefficient.

        Defaults to 1.0.
    cso_repeat_num : int, optional
        Number of repetitions to run CSO. CSO starts with a different initializaiton at each time. Used in CSO.

        Defaults to 2.
    cso_maxgeneration_num : int, optional
        Maximal number of generations. Used in CSO.

        Defaults to 100.
    cso_earlystop_num : int, optional
        Stop if there's no change in generation. Used in CSO.

        Defaults to 30.
    cso_population_size : int, optional
        Population size of the swarm particles. Used in CSO.

        Defaults to 30.
    cso_phi : float, optional
        Social factor. Used in CSO.

        Defaults to 0.1.
    cso_featurenum_penalty : float, optional
        The ratio for the spliting of training data and testing data.

        Defaults to 0.1.
    cso_test_ratio : float, optional
        The ratio for the spliting of training data and testing data.

        Defaults to 0.2.

    Attributes
    ----------
    result_ : DataFrame
        PAL returned result, structured as follows:
         - ROWID: Indicates the id of current row.
         - OUTPUT: Best set of features.

    Examples
    --------
    Original data:

    >>> df.collect()
       X1 X2    X3    X4 X5 X6 X7    X8 X9 X10 X11 X12 X13 Y
    0  1  22.08 11.46 2  4  4  1.585 0  0  0   1   2   100 1,213
    1  0  22.67 7     2  8  4  0.165 0  0  0   0   2   160 1
    2  0  29.58 1.75  1  4  4  1.25  0  0  0   1   2   280 1
    3  0  21.67 11.5  1  5  3  0     1  1  11  1   2   0   1
    4  1  20.17 8.17  2  6  4  1.96  1  1  14  0   2   60  159
    5  0  15.83 0.585 2  8  8  1.5   1  1  2   0   2   100 1
    6  1  17.42 6.5   2  3  4  0.125 0  0  0   0   2   60  101
    7  0  58.67 4.46  2  11 8  3.04  1  1  6   0   2   43  561
    8  1  27.83 1     1  2  8  3     0  0  0   0   2   176 538
    9  0  55.75 7.08  2  4  8  6.75  1  1  3   1   2   100 51
    10 1  33.5  1.75  2  14 8  4.5   1  1  4   1   2   253 858
    11 1  41.42 5     2  11 8  5     1  1  6   1   2   470 1
    12 1  20.67 1.25  1  8  8  1.375 1  1  3   1   2   140 211
    13 1  34.92 5     2  14 8  7.5   1  1  6   1   2   0   1,001
    14 1  58.58 2.71  2  8  4  2.415 0  0  0   1   2   320 1
    15 1  48.08 6.04  2  4  4  0.04  0  0  0   0   2   0   2,691
    16 1  29.58 4.5   2  9  4  7.5   1  1  2   1   2   330 1
    17 0  18.92 9     2  6  4  0.75  1  1  2   0   2   88  592
    18 1  20    1.25  1  4  4  0.125 0  0  0   0   2   140 5
    19 0  22.42 5.665 2  11 4  2.585 1  1  7   0   2   129 3,258
    20 0  28.17 0.585 2  6  4  0.04  0  0  0   0   2   260 1,005
    21 0  19.17 0.585 1  6  4  0.585 1  0  0   1   2   160 1
    22 1  41.17 1.335 2  2  4  0.165 0  0  0   0   2   168 1
    23 1  41.58 1.75  2  4  4  0.21  1  0  0   0   2   160 1
    24 1  19.5  9.585 2  6  4  0.79  0  0  0   0   2   80  351
    25 1  32.75 1.5   2  13 8  5.5   1  1  3   1   2   0   1
    26 1  22.5  0.125 1  4  4  0.125 0  0  0   0   2   200 71
    27 1  33.17 3.04  1  8  8  2.04  1  1  1   1   2   180 18,028
    28 0  30.67 12    2  8  4  2     1  1  1   0   2   220 20
    29 1  23.08 2.5   2  8  4  1.085 1  1  11  1   2   60  2,185

    Construct an Discretize instance:

    >>> fs = FeatureSelection(fs_method='fisher-score',
                              top_k_best=8)
    >>> fs_df = fs.fit_transform(df,
                                 categorical_variable=['X1'],
                                 label='Y')
    >>> fs.result_.collect()
      ROWID                                                                                              OUTPUT
    0     0     {"__method__":"fisher-score","__SelectedFeatures__":["X3","X7","X2","X8","X9","X13","X6","X5"]}
    >>> fs_df.collect()
       X3    X7    X2    X8 X9 X13 X6 X5
    0  11.46 1.585 22.08 0  0  100 4  4
    1  7     0.165 22.67 0  0  160 4  8
    2  1.75  1.25  29.58 0  0  280 4  4
    3  11.5  0     21.67 1  1  0   3  5
    4  8.17  1.96  20.17 1  1  60  4  6
    5  0.585 1.5   15.83 1  1  100 8  8
    6  6.5   0.125 17.42 0  0  60  4  3
    7  4.46  3.04  58.67 1  1  43  8  11
    8  1     3     27.83 0  0  176 8  2
    9  7.08  6.75  55.75 1  1  100 8  4
    10 1.75  4.5   33.5  1  1  253 8  14
    11 5     5     41.42 1  1  470 8  11
    12 1.25  1.375 20.67 1  1  140 8  8
    13 5     7.5   34.92 1  1  0   8  14
    14 2.71  2.415 58.58 0  0  320 4  8
    15 6.04  0.04  48.08 0  0  0   4  4
    16 4.5   7.5   29.58 1  1  330 4  9
    17 9     0.75  18.92 1  1  88  4  6
    18 1.25  0.125 20    0  0  140 4  4
    19 5.665 2.585 22.42 1  1  129 4  11
    20 0.585 0.04  28.17 0  0  260 4  6
    21 0.585 0.585 19.17 1  0  160 4  6
    22 1.335 0.165 41.17 0  0  168 4  2
    23 1.75  0.21  41.58 1  0  160 4  4
    24 9.585 0.79  19.5  0  0  80  4  6
    25 1.5   5.5   32.75 1  1  0   8  13
    26 0.125 0.125 22.5  0  0  200 4  4
    27 3.04  2.04  33.17 1  1  180 8  8
    28 12    2     30.67 1  1  220 4  8
    29 2.5   1.085 23.08 1  1  60  4  8
    """
    def __init__(self,#pylint: disable=too-many-arguments
                 fs_method,
                 top_k_best=None,
                 thread_ratio=None,
                 seed=None,
                 fs_threshold=None,
                 fs_n_neighbours=None,
                 fs_category_weight=None,
                 fs_sigma=None,
                 fs_regularization_power=None,
                 fs_rowsampling_ratio=None,
                 fs_max_iter=None,
                 fs_admm_tol=None,
                 fs_admm_rho=None,
                 fs_admm_mu=None,
                 fs_admm_gamma=None,
                 cso_repeat_num=None,
                 cso_maxgeneration_num=None,
                 cso_earlystop_num=None,
                 cso_population_size=None,
                 cso_phi=None,
                 cso_featurenum_penalty=None,
                 cso_test_ratio=None
                 ):
        setattr(self, 'hanaml_parameters', pal_param_register())
        super(FeatureSelection, self).__init__()
        self.method_map = {'anova': 0,
                           'chi-squared': 1,
                           'gini-index': 2,
                           'fisher-score': 3,
                           'information-gain': 4,
                           'mrmr': 5,
                           'jmi': 6,
                           'iwfs': 7,
                           'fcbf': 8,
                           'laplacian-score': 9,
                           'spec': 10,
                           'relieff': 11,
                           'admm': 12,
                           'cso': 13}
        self.fs_method = self._arg('fs_method', fs_method, self.method_map, required=True)
        if self.fs_method not in (8, 13):
            self.top_k_best = self._arg('top_k_best', top_k_best, int, required=True)
        else:
            self.top_k_best = self._arg('top_k_best', top_k_best, int)
        self.thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        self.seed = self._arg('seed', seed, float)
        self.fs_threshold = self._arg('fs_threshold', fs_threshold, float)
        self.fs_n_neighbours = self._arg('fs_n_neighbours', fs_n_neighbours, int)
        self.fs_category_weight = self._arg('fs_category_weight', fs_category_weight, float)
        self.fs_sigma = self._arg('fs_sigma', fs_sigma, float)
        self.fs_regularization_power = self._arg('fs_regularization_power', fs_regularization_power, int)
        self.fs_rowsampling_ratio = self._arg('fs_rowsampling_ratio', fs_rowsampling_ratio, float)
        self.fs_max_iter = self._arg('fs_max_iter', fs_max_iter, int)
        self.fs_admm_tol = self._arg('fs_admm_tol', fs_admm_tol, float)
        self.fs_admm_rho = self._arg('fs_admm_rho', fs_admm_rho, float)
        self.fs_admm_mu = self._arg('fs_admm_mu', fs_admm_mu, float)
        self.fs_admm_gamma = self._arg('fs_admm_gamma', fs_admm_gamma, float)
        self.cso_repeat_num = self._arg('cso_repeat_num', cso_repeat_num, int)
        self.cso_maxgeneration_num = self._arg('cso_maxgeneration_num', cso_maxgeneration_num, int)
        self.cso_earlystop_num = self._arg('cso_earlystop_num', cso_earlystop_num, int)
        self.cso_population_size = self._arg('cso_population_size', cso_population_size, float)
        self.cso_phi = self._arg('cso_phi', cso_phi, float)
        self.cso_featurenum_penalty = self._arg('cso_featurenum_penalty', cso_featurenum_penalty, float)
        self.cso_test_ratio = self._arg('cso_test_ratio', cso_test_ratio, float)
        self.result_ = None

    def fit_transform(self, data,
                      key=None,
                      label=None,#pylint:disable=too-many-arguments
                      categorical_variable=None,
                      fixed_feature=None,
                      excluded_feature=None,
                      verbose=None):
        """
        Upsamping given datasets using SMOTE with specified configuration.

        Parameters
        ----------
        data : DataFrame
             Dataframe that contains the training data.
        key : str, optional
            Name of the ID column. If data has index, it will be set.

            There's no id column by default.
        label : str
            Specifies the dependent variable by name.
        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.
        fixed_feature : str or list of str, optional
            Will always be selected out as the best subset.
        excluded_feature : str or list of str, optional
            Excludes the indicated columns as feature candidates.
        verbose : bool, optional
            Indicates whether to output more specified results.

            Defaults to False.

        Returns
        -------
        DataFrame
             - Feature selected result from the input data.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        index = data.index
        has_id = False
        if key:
            has_id = True
        if index:
            has_id = True
        if has_id:
            key = self._arg('key', key, str, required=not isinstance(index, str))
            if isinstance(index, str):
                if key is not None and index != key:
                    msg = "Discrepancy between the designated key column '{}' ".format(key) +\
                    "and the designated index column '{}'.".format(index)
                    logger.warning(msg)
            key = index if key is None else key
            cols = data.columns
            cols.remove(key)
            data = data[[key] + cols]
        label = arg('label', label, str)
        if isinstance(fixed_feature, str):
            fixed_feature = [fixed_feature]
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        if isinstance(excluded_feature, str):
            excluded_feature = [excluded_feature]
        fixed_feature = self._arg('fixed_feature', fixed_feature, ListOfStrings)
        categorical_variable = self._arg('categorical_variable', categorical_variable,
                                         ListOfStrings)
        excluded_feature = self._arg('excluded_feature', excluded_feature, ListOfStrings)
        verbose = self._arg('verbose', verbose, bool)
        tables = ['RESULT']
        tables = ['#FEATURESELECTION_{}_TBL_{}'.format(tbl, unique_id) for tbl in tables]
        res_tbl = tables[0]

        param_rows = [('FS_METHOD', self.fs_method, None, None),
                      ('HAS_ID', has_id, None, None),
                      ('TOP_K_BEST', self.top_k_best, None, None),
                      ('DEPENDENT_VARIABLE', None, None, label),
                      ('VERBOSE', verbose, None, None),
                      ('THREAD_RATIO', None, self.thread_ratio, None),
                      ('SEED', self.seed, None, None),
                      ('FS_ROWSAMPLING_RATIO', None, self.fs_rowsampling_ratio, None),
                      ('FS_THRESHOLD', None, self.fs_threshold, None),
                      ('FS_N_NEIGHBOURS', self.fs_n_neighbours, None, None),
                      ('FS_CATEGORY_WEIGHT', None, self.fs_category_weight, None),
                      ('FS_SIGMA', None, self.fs_sigma, None),
                      ('FS_REGULARIZATION_POWER', self.fs_regularization_power, None, None),
                      ('FS_MAX_ITER', self.fs_max_iter, None, None),
                      ('FS_ADMM_TOL', None, self.fs_admm_tol, None),
                      ('FS_ADMM_RHO', None, self.fs_admm_rho, None),
                      ('FS_ADMM_MU', None, self.fs_admm_mu, None),
                      ('FS_ADMM_GAMMA', None, self.fs_admm_gamma, None),
                      ('CSO_REPEAT_NUM', self.cso_repeat_num, None, None),
                      ('CSO_MAXGENERATION_NUM', self.cso_maxgeneration_num, None, None),
                      ('CSO_EARLYSTOP_NUM', self.cso_earlystop_num, None, None),
                      ('CSO_POPULATION_SIZE', self.cso_population_size, None, None),
                      ('CSO_PHI', None, self.cso_phi, None),
                      ('CSO_FEATURENUM_PENALTY', None, self.cso_featurenum_penalty, None),
                      ('CSO_TEST_RATIO', None, self.cso_test_ratio, None)]
        if fixed_feature is not None:
            param_rows.extend([('FIXED_FEATURE', None,
                                None, var) for var in fixed_feature])
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None,
                                None, var) for var in categorical_variable])
        if excluded_feature is not None:
            param_rows.extend([('EXCLUDED_FEATURE', None,
                                None, var) for var in excluded_feature])
        try:
            self._call_pal_auto(conn,
                                "PAL_FEATURE_SELECTION",
                                data,
                                ParameterTable().with_data(param_rows),
                                *tables)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        self.result_ = conn.table(res_tbl)
        select_cols = json.loads(self.result_.collect().iat[0, 1])["__SelectedFeatures__"]
        if has_id:
            if key not in select_cols:
                select_cols = [key] + select_cols
        return data.select(select_cols)
