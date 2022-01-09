"""
This module contains Python wrapper for PAL fast dtw algorithm.

The following function is available:

    * :func:`dtw`

"""

#pylint:disable=line-too-long, too-many-arguments, too-many-locals
#pylint: disable=consider-using-f-string
import logging
import uuid
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
    ListOfTuples,
    call_pal_auto_with_hint
)

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

def dtw(query_data,#pylint:disable=too-many-arguments, too-few-public-methods, too-many-locals
        ref_data,
        radius=None,
        thread_ratio=None,
        distance_method=None,
        minkowski_power=None,
        alignment_method=None,
        step_pattern=None,
        save_alignment=None):
    r"""
    DTW is an abbreviation for Dynamic Time Warping. It is a method for calculating distance or similarity between two time series.
    It makes one series match the other one as much as possible by stretching or compressing one or both two.

    Note that this function is a new function in SAP HANA Cloud QRC01/2021.

    Parameters
    ----------

    query_data : DataFrame
        Query data for DTW, expected to be structured as follows:

            - 1st column : ID of query time-series, type INTEGER, VARCHAR or NVARCHAR.
            - 2nd column : Order(timestamps) of query time-series, type INTEGER, VARCHAR or NVARCHAR.
            - Other columns : Series data, type INTEGER, DOUBLE or DECIMAL.

    ref_data : DataFrame
        Reference data for DTW, expected to be structed as follows:

            - 1st column : ID of reference time-series, type INTEGER, VARCHAR or NVARCHAR
            - 2nd column : Order(timestamps) of reference time-series, type INTEGER, VARCHAR or NVARCHAR
            - Other columns : Series data, type INTEGER, DOUBLE or DECIMAL, must have the same cardinality(i.e. number of columns)
              as that of ``data``.

    radius : int, optional
        Specifies a constraint to restrict match curve in an area near diagonal.

        To be specific, it makes sure that the absolute difference for each pair of
        subscripts in the match curve is no greater than ``radius``.

        -1 means no such constraint, otherwise ``radius`` must be nonnegative.

        Defaults to -1.

    thread_ratio : float, optional
        Controls the proportion of available threads to use.
        The ratio of available threads.

            - 0 : single thread.
            - 0~1 : percentage.
            - Others : heuristically determined.

        Defaults to -1.

    distance_method : {'manhattan', 'euclidean', 'minkowski', 'chebyshev', 'cosine'}, optional

        Specifies the method to compute the distance between two points.

            - 'manhattan' : Manhattan distance
            - 'euclidean' : Euclidean distance
            - 'minkowski' : Minkowski distance
            - 'chebyshev' : Chebyshev distance
            - 'cosine' : Cosine distance

        Defaults to 'euclidean'.

    minkowski_power : double, optional
        Specifies the power of the Minkowski distance method.

        Only valid when ``distance_method`` is 'minkowski'.

        Defaults to 3.

    alignment_method : {'closed', 'open_begin', 'open_end', 'open'}
        Specifies the alignment constraint w.r.t. beginning and end points in reference time-series.

            - 'closed' : Both begining and end points must be aligned.
            - 'open_end' : Only beginning point needs to be aligned.
            - 'open_begin': Only end point needs to be aligned.
            - 'open': Neither beginning nor end point need to be aligned.

        Defaults to 'closed'.

    step_pattern : int or ListOfTuples
        Specifies the type of step patterns for DTW algorithm.

        There are five predefined types of step patterns, ranging from 1 to 5.

        Users can also specify custom defined step patterns by providing a list tuples.

        Defaults to 3.

        .. note::
            A custom defined step pattern is reprenseted either by a single triad or a tuple of consecutive triads, where
            each triad is in the form of :math:`(\Delta x, \Delta y, \omega)` with :math:`\Delta x` being the increment in
            query data index, :math:`\Delta y` being the increment in reference data index, and :math:`\omega` being the weight.

            A custom defined step pattern type is simply a list of steps patterns.

            For example, the predefined step patterns of type 5 can also be specified via custom defined
            step pattern type as follows:

            [((1,1,1), (1,0,1)), (1,1,1), ((1,1,0.5), (0,1,0.5))].

            For more details on step patterns, one may go to
            `PAL DTW`_ for reference.

            .. _PAL DTW: https://help.sap.com/viewer/319d36de4fd64ac3afbf91b1fb3ce8de/2021_01_QRC/en-US/2b949ae44191490b8a89261ed2f21728.html

    save_alignment : bool, optional
        Specifies whether to output alignment information or not.

            - True : Ouput the alignment information.
            - False : Do not output the alignment information.

        Defaults to False.

    Returns
    -------
    DataFrames

        - Result for DTW, structured as follows:

            - QUERY_<ID column name of query data table> : ID of the query time-series.
            - REF_<ID column name of reference data table> : ID of the reference time-series.
            - DISTANCE : DTW distance of the two series. NULL if there is no valid result.
            - WEIGHT : Total weight of match.
            - AVG_DISTANCE : Normalized distance of two time-series. NULL if WEIGHT is near 0.

        - Alignment information table, structured as follows:

            - QUERY_<ID column name of query data table> : ID of query time-series.
            - REF_<ID column name of input table> : ID of reference time-series.
            - QUERY_INDEX : Corresponding to index of query time-series.
            - REF_INDEX : Corresponding to index of reference time-series.

        - Statistics for DTW, structured as follows:

            - STAT_NAME: Statistics name.
            - STAT_VALUE: Statistics value.

    Examples
    --------
    Query data:

    >>> data1.collect()
       ID  TIMESTAMP  ATTR1  ATTR2
    0   1          1      1    5.2
    1   1          2      2    5.1
    2   1          3      3    2.0
    3   1          4      4    0.3
    4   1          5      5    1.2
    5   1          6      6    7.7
    6   1          7      7    0.0
    7   1          8      8    1.1
    8   1          9      9    3.2
    9   1         10     10    2.3
    10  2          1      7    2.0
    11  2          2      6    1.4
    12  2          3      1    0.9
    13  2          4      3    1.2
    14  2          5      2   10.2
    15  2          6      5    2.3
    16  2          7      4    4.5
    17  2          8      3    4.6
    18  2          9      3    3.5

    Reference data:

    >>> data2.collect()
       ID  TIMESTAMP  ATTR1  ATTR2
    0   3          1     10    1.0
    1   3          2      5    2.0
    2   3          3      2    3.0
    3   3          4      8    1.4
    4   3          5      1   10.8
    5   3          6      5    7.7
    6   3          7      5    6.3
    7   3          8     12    2.4
    8   3          9     20    9.4
    9   3         10      4    0.5
    10  3         11      6    2.2

    Call dtw():

    >>> res, align, stats = dtw(data1, data2,
    ...                         step_pattern=[((1,1,1),(1,0,1)), (1,1,1), ((1,1,0.5),(0,1,0.5))],
    ...                         save_alignment=True)
    >>> res.collect()
      QUERY_ID REF_ID   DISTANCE  WEIGHT  AVG_DISTANCE
    0        1      3  48.027427    10.0      4.802743
    1        2      3  36.933929     9.0      4.103770
    """
    conn = query_data.connection_context
    require_pal_usable(conn)
    distance_map = dict(manhattan=1, euclidean=2, minkowski=3, chebyshev=4, cosine=6)
    alignment_map = dict(closed='CLOSED', open_end='OPEN_END',
                         open_begin='OPEN_BEGIN', open='OPEN_BEGIN_END')
    radius = arg('radius', radius, int)
    thread_ratio = arg('thread_ratio', thread_ratio, float)
    distance_method = arg('distance_method', distance_method, distance_map)
    #if isinstance(distance_method, str):
    #    distance_method = arg('distance_method', distance_method, distance_map)
    alignment_method = arg('alignment_method', alignment_method, alignment_map)
    if not isinstance(step_pattern, int):
        step_pattern = arg('step_pattern', step_pattern, ListOfTuples)
    pattern_type = step_pattern if isinstance(step_pattern, int) else None
    minkowski_power = arg('minkowski_power', minkowski_power, float)
    save_alignment = arg('save_alignment', save_alignment, bool)

    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    outputs = ['RESULT', 'ALIGNMENT', 'STATS']
    outputs = ['#PAL_DTW_{}_TBL_{}_{}'.format(name, id, unique_id) for name in outputs]
    res_tbl, align_tbl, stats_tbl = outputs
    patterns = None
    param_rows = [('WINDOW', radius, None, None),
                  ('THREAD_RATIO', None, thread_ratio, None),
                  ('DISTANCE_METHOD', distance_method, None, None),
                  ('STEP_PATTERN_TYPE', pattern_type, None, None),
                  ('ALIGNMENT_BEGIN_END', None, None, alignment_method),
                  ('MINKOWSKI_POWER', None, minkowski_power, None),
                  ('SAVE_ALIGNMENT', save_alignment, None, None)]
    if step_pattern is not None and pattern_type is None:
        patterns = [str(s).replace('((', '(').replace('))', ')') for s in step_pattern] if \
        isinstance(step_pattern, list) else None
        param_rows.extend([('STEP_PATTERN', None, None, pattern) for pattern in patterns])
    try:
        call_pal_auto_with_hint(conn,
                                None,
                                'PAL_DTW',
                                query_data,
                                ref_data,
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
    return conn.table(res_tbl), conn.table(align_tbl), conn.table(stats_tbl)
