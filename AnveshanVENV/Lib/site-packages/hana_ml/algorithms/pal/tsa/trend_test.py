"""
This module contains Python wrapper for PAL trend test algorithm.

The following function is available:

    * :func:`trend_test`
"""
#pylint:disable=line-too-long, too-many-arguments, too-few-public-methods
#pylint: disable=invalid-name, unused-argument, too-many-locals, too-many-statements
#pylint: disable=attribute-defined-outside-init, unused-variable
#pylint: disable=consider-using-f-string
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

def trend_test(data, key=None, endog=None, method=None, alpha=None):
    r"""
    Trend test is able to identify whether a time series has an upward or downward trend or not, and calculate the de-trended time series.

    Parameters
    ----------

    data : DataFrame

        Input data. At least two columns, one is ID column, the other is raw data.

    key : str, optional
        The ID column.

        Defaults to the first column of data if the index column of data is not provided.
        Otherwise, defaults to the index column of data.

    endog : str, optional
        The column of series to be tested.

        Defaults to the first non-ID column.

    method : {'mk', 'difference-sign'}, optional
        The method used to identify trend:

            'mk': Mann-Kendall test.

            'difference-sign': Difference-sign test.

        Defaults to 'mk'.

    alpha : float, optional
        Significance value.

        The value range is (0, 0.5).

        Defaults to 0.05.

    Returns
    -------

    DataFrame

        Statistics for time series, structured as follows:
            - STAT_NAME: includes

              - TREND: -1 for downward trend, 0 for no trend, and 1 for upward trend
              - S: the number of positive pairs minus the negative pairs
              - P-VALUE: The p-value of the observed S

            - STAT_VALUE: value of stats above.

        Detrended table, structured as follows:
            - ID : Time stamp that is monotonically increasing sorted.
            - DETRENDED_SERIES: The corresponding de-trended time series. The first value absents if trend presents.


    Examples
    --------

    Time series data df:

    >>> df.collect().head()
           TIME_STAMP  SERIES
    0      1           1500
    1      2           1510
    2      3           1550

    Perform trend_test function:

    >>> stats, detrended = trend_test(data=df, key='TIME_STAMP', endog='SERIES', method='mk', alpha=0.05)

    Outputs:

    >>> stats.collect()
         STAT_NAME        STAT_VALUE
    0    TREND            1
    1    S                60
    2    P-VALUE          0.0000267...

    >>> detrended.collect().head(2)
         ID    DETRENDED_SEARIES
    1    2     10
    2    3     40

    """
    conn = data.connection_context
    require_pal_usable(conn)
    method_map = {'mk':1, 'difference-sign':2}

    method = arg('method', method, method_map)
    alpha = arg('alpha', alpha, float)
    key = arg('key', key, str)
    endog = arg('endog', endog, str)

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

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = ['STATS', 'DETRENDED',]
    outputs = ['#PAL_TREND_TEST_{}_TBL_{}_{}'.format(name, id, unique_id) for name in outputs]
    stats_tbl, detrended_tbl = outputs

    param_rows = [('METHOD', method, None, None),
                  ('ALPHA', None, alpha, None)]
    try:
        call_pal_auto_with_hint(conn,
                                None,
                                'PAL_TREND_TEST',
                                data_,
                                ParameterTable().with_data(param_rows),
                                *outputs)

    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn, stats_tbl)
        try_drop(conn, detrended_tbl)
        raise
    except pyodbc.Error as db_err:
        logger.exception(str(db_err.args[1]))
        try_drop(conn, stats_tbl)
        try_drop(conn, detrended_tbl)
        raise
    return conn.table(stats_tbl), conn.table(detrended_tbl)
