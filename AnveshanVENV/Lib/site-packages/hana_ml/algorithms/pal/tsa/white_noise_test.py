"""
This module contains Python wrapper for PAL white noise test algorithm.

The following function is available:

    * :func:`white_noise_test`
"""
#pylint:disable=line-too-long, attribute-defined-outside-init, unused-variable, too-many-arguments
#pylint:disable=invalid-name, too-few-public-methods, too-many-statements, too-many-locals
#pylint:disable=consider-using-f-string
import logging
import uuid
import warnings
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    arg,
    try_drop,
    require_pal_usable,
    call_pal_auto_with_hint
)
logger = logging.getLogger(__name__)

def white_noise_test(data, key=None, endog=None, lag=None, probability=None, thread_ratio=None, model_df=None):
    r"""
    This algorithm is used to identify whether a time series is a white noise series.
    If white noise exists in the raw time series, the algorithm returns the value of 1. If not, the value of 0 will be returned.

    Parameters
    ----------

    data : DataFrame
        Input data which contains at least two columns, one is ID column, the other is raw data.

    key : str, optional
        The ID column.

        Defaults to the first column of data if the index column of data is not provided.
        Otherwise, defaults to the index column of data.

    endog : str, optional

        The column of series to be tested.

        Defaults to the first non-key column.

    lag : int, optional
        Specifies the lag autocorrelation coefficient that the statistic will be based on.

        It corresponds to the degree of freedom of chi-square distribution.

        Defaults to half of the sample size (n/2).

    probability : float, optional
        The confidence level used for chi-square distribution.

        The value is 1 - a, where a is the significance level.

        Defaults to 0.9.

    thread_ratio : float, optional
        The ratio of available threads.
            - 0: single thread.
            - 0~1: percentage.
            - Others: heuristically determined.

        Defaults to -1.

    model_df : int, optional
        Specifies the number of degrees of freedom occuppied by a model.

        Should be provided if the input data is the residual of some raw time-series data
        after being fitted by a model.

        Defaults to 0.

    Returns
    -------
    DataFrame
        Statistics for time series, structured as follows:
            - STAT_NAME: Name of the statistics of the series.
            - STAT_VALUE: include following values:
                    - WN: 1 for white noise, 0 for not white noise.
                    - Q: Q statistics defined as above.
                    - chi^2: chi-square distribution.

    Examples
    --------

    Time series data df:

    >>> df.collect().head(3)
           TIME_STAMP  SERIES
    0      1           1356.00
    1      2           826.00
    2      3           1586.00

    Perform white_noise_test function:

    >>> stats = white_noise_test(data=df, endog='SERIES', lag=3, probability=0.9, thread_ratio=0.2)

    Outputs:

    >>> stats.collect()
         STAT_NAME     STAT_VALUE
    0    WN            1.000000
    1    Q             5.576053
    2    chi^2         6.251389
    """
    conn = data.connection_context
    require_pal_usable(conn)
    lag = arg('method', lag, int)
    probability = arg('probability', probability, float)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    model_df = arg('model_df', model_df, int)
    key = arg('key', key, str)
    endog = arg('endog', endog, str)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    stats_tbl = '#PAL_WHITE_NOISE_TEST_STATS_TBL_{}_{}'.format(id, unique_id)

    cols = data.columns
    if len(cols) < 2:
        msg = ("Input data should contain at least 2 columns: " +
               "one for ID, another for raw data.")
        logger.error(msg)
        raise ValueError(msg)

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

    if endog is not None:
        if endog not in cols:
            msg = ('Please select endog from name of columns!')
            logger.error(msg)
            raise ValueError(msg)
    else:
        endog = cols[0]

    data_ = data[[key] + [endog]]

    param_rows = [('LAG', lag, None, None),
                  ('PROBABILITY', None, probability, None),
                  ('THREAD_RATIO', None, thread_ratio, None),
                  ('MODEL_DF', model_df, None, None)]

    try:
        call_pal_auto_with_hint(conn,
                                None,
                                'PAL_WN_TEST',
                                data_,
                                ParameterTable().with_data(param_rows),
                                stats_tbl)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, stats_tbl)
        raise
    except pyodbc.Error as db_err:
        logger.exception(str(db_err.args[1]))
        try_drop(conn, stats_tbl)
        raise
    return conn.table(stats_tbl)
