"""
This module contains Python wrappers for PAL Discrete Wavelet Transform and
PAL Discrete Wavelet Packet Transform.

The following classes and functions are available:

    * :class:`DWT`
    * :func:`wavedec`
    * :func:`waverec`
    * :func:`wpdec`
    * :func:`wprec`

"""
#pylint:disable=line-too-long, too-many-locals, attribute-defined-outside-init, unused-argument
#pylint:disable=invalid-name, too-many-arguments, too-few-public-methods, too-many-statements
#pylint: disable=consider-using-f-string
import logging
import uuid
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.dataframe import DataFrame
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    arg,
    try_drop,
    require_pal_usable
)
logger = logging.getLogger(__name__)

class DWT(PALBase):
    r"""
    A designed class for discrete wavelet transform and wavelet packet transform.

    Parameters
    ----------
    wavelet : str
        Specifies the wavelet filter used for discrete wavelet transform. Valid options include:

            - Daubechies family : 'db1' ~ 'db20'
            - Biorthognal family: 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',
              'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
              'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'
            - Reverse Biorthogal family : 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2',
              'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',
              'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'
            - Coifman family: 'coif1' ~ 'coif5'
            - Symmetric family: 'sym2' ~ 'sym20'

    boundary : str, otpional
        Specifies the padding method for boundary values. Valid options include:

            - 'zero' : Zero padding
            - 'symmetric' : Symmetric padding
            - 'periodic' : Periodic padding
            - 'reflect' : Reflect padding
            - 'smooth' : Smooth padding
            - 'constant' : Constant padding

        Defaults to 'zero'.

    level : int, optional
        Specifies the decompose level for wavelet (packet) transform.

        Defaults to 1.

    packet : bool, optional
        Specifies whether or not to perform wavelet packet transformation.

            - True : to perform wavelet packet transformation.
            - False : to perform discrete wavelet transform.

        Defaults to False.

    order : {'index', 'frequency'}
       Specifies the order of node in the wavelet packet coefficients table.

           - 'index' : ordered by the indices of nodes(ascending).
           - 'frequency' : ordered by the frequencies of nodes, from low to high.

        Valid only when ``packet`` is True.

        Defaults to 'index'.

    Attributes
    ----------
    coeff\_ : DataFrame
        DataFrame containing the result of PAL multi-level discrete wavelet (packet) transformation.

        If ``packet`` is False, then the DataFrame is expected to be structured as follows:

            - 1st column : LEVEL, representing the decomposition level of coefficients, where
              approximation coefficients are marked with 0 while details coefficients are marked by
              their respective levels.
            - 2nd column : ID, representing the order of coefficients.
            - 3rd column : Approximation and detail coefficients for discrete wavelet transform(decomposition).

        If ``packet`` is True, the the DataFrame is expected to be structured as follows:
            - 1st column : NODE, representing index of the nodes(i.e. leaves of the binary tree
              for wavelet packet transformation).
            - 2nd column : ID, representing the time/spatial order of coefficients.
            - 3rd column : COEFFICIENTS, wavelet packet coefficients.

    stats\_ : DataFrame
        DataFrame containing the key statistics for multi-level wavelet packet transformation.

        The DataFrame is expected to be structured as follows:

            - 1st column : NAME, type NVARCHAR(100), names of statistics.
            - 2nd column : VAL, type NVARCHAR(5000), value of statistics(in particular, the
              coefficient size in different decomposition levels).

        Avalible only when ``packet`` is True.

    """
    boundaries = ['zero', 'symmetric', 'periodic', 'reflect', 'smooth', 'constant']
    filters = ['db' + str(i) for i in range(1, 21)] + \
    ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',\
     'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',\
     'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'] + \
    ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2',\
     'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',\
     'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'] +\
    ['coif' + str(i) for i in range(1, 6)] + \
    ['sym' + str(i) for i in range(2, 21)]
    orders = ['index', 'frequency']
    def __init__(self,
                 wavelet,
                 boundary='zero',
                 level=None,
                 packet=False,
                 order=None):
        super(DWT, self).__init__()
        self.wavelet = self._arg('wavelet', wavelet, {filter:filter for filter in self.filters},
                                 required=True)
        self.boundary = self._arg('boundary', boundary,
                                  {bd:idx for idx, bd in enumerate(self.boundaries)})
        self.level = self._arg('level', level, int)
        self.packet = self._arg('packet', packet, bool)
        self.order = self._arg('order', order, {od:idx for idx, od in enumerate(self.orders)})
        self.coeff_ = None
        self.stats_ = None

    def transform(self, data, key, col=None):
        r"""
        Performing the forward transformation(i.e. decomposition) for discrete wavelet transformation or
        wavelet packet transformation.

        Parameters
        ----------
        data : DataFrame
            Time-series data to apply discrete wavelet transform to.

        key : str
            Specifies the time-stamp column in ``data``.

            The column should be of type INTEGER, and the values do not need to be arranged in order.

        col : str, optional
            Specifies the signal values for wavelet transformation, should be of type DOUBLE
            or DECIMAL(p,s).

            The values contained in this column must be evenly-spaced in time.

            If not specified, it defaults to the first non-key column of ``data``.

        Returns
        -------
        A `DTW` object with wavelet coefficients as its attribute.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        if len(cols) < 2:
            msg = 'Insufficient number of columns in `data`: ' + \
            'at least 2 columns are required.'
            raise ValueError(msg)
        key = self._arg('key', key, str, required=True)
        key = self._arg('key', key, {col.lower():col for col in cols})
        cols.remove(key)
        col = self._arg('col', col, str)
        col = self._arg('col', col, {col.lower():col for col in cols})
        if col is None:
            col = cols[0]
        data_ = data[[key, col]]
        param_rows = [('FILTER_TYPE', None, None, self.wavelet),
                      ('PADDING_TYPE', self.boundary, None, None),
                      ('LEVEL', self.level, None, None)]
        if self.packet is True:
            param_rows.extend([('OUT_TYPE', self.order, None, None)])
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        pal_proc = 'WAVELET_PACKET_TRANSFORM' if self.packet else 'DISCRETE_WAVELET_TRANSFORM'
        result_tbl = '#PAL_{}_RESULT_TBL_{}'.format(pal_proc, unique_id)
        if self.packet is True:
            stats_tbl = '#PAL_{}_STATS_TBL_{}'.format(pal_proc, unique_id)
            tables = [result_tbl, stats_tbl]
        else:
            tables = [result_tbl]
        try:
            self._call_pal_auto(conn,
                                'PAL_' + pal_proc,
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
        self.coeff_ = conn.table(result_tbl)
        if self.packet is True:
            self.stats_ = conn.table(stats_tbl)
        return self

    def inverse(self, wavelet=None, boundary=None):
        r"""
        Inverse transformation of wavelet decomposition, i.e. reconstruction.

        Parameters
        ----------
        wavelet : str, optional
            Specifies the wavelet filter used in the decomposition phase.

            If not provided, the value in self.wavelet is used.

        boundary : str, optional
            Specifies the padding method for boundary values. Valid optionals include:
            'zero', 'symmetric', 'periodic', 'reflect', 'smooth' and 'constant'.

            If not provided, the value in self.boundary is used.

        Returns
        -------
        DataFrame
           The reconstructed time-series data after inverse transformation.
        """
        if self.coeff_ is None:
            msg = 'Coefficients not available. ' +\
            'Need to call transform() firstly for the input DWT object.'
            raise TypeError(msg)
        if self.stats_ is None and self.packet is True:
            msg = 'Statistics table is missing for inverse wavelet packet transformation.'
            raise TypeError(msg)
        data = self.coeff_
        conn = data.connection_context
        require_pal_usable(conn)
        wavelet = self._arg('wavelet', wavelet, {filter:filter for filter in self.filters})
        boundary = self._arg('boundary', boundary,
                             {bd:idx for idx, bd in enumerate(self.boundaries)})
        wavelet = self.wavelet if wavelet is None else wavelet
        boundary = self.boundary if boundary is None else boundary
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        pal_proc = 'WAVELET_PACKET_TRANSFORM_INVERSE' if self.packet else \
        'DISCRETE_WAVELET_TRANSFORM_INVERSE'
        result_tbl = '#PAL_{}_RESULT_TBL_{}'.format(pal_proc, unique_id)
        param_rows = [('FILTER_TYPE', None, None, wavelet),
                      ('PADDING_TYPE', boundary, None, None)]
        try:
            if self.packet:
                self._call_pal_auto(conn,
                                    'PAL_' + pal_proc,
                                    self.coeff_,
                                    self.stats_,
                                    ParameterTable().with_data(param_rows),
                                    *[result_tbl])
            else:
                self._call_pal_auto(conn,
                                    'PAL_' + pal_proc,
                                    self.coeff_,
                                    ParameterTable().with_data(param_rows),
                                    *[result_tbl])
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, result_tbl)
            raise
        return conn.table(result_tbl)

def wavedec(data, wavelet, key=None, col=None, boundary=None, level=None):
    r"""
    Python wrapper for PAL multi-level discrete wavelet transform.

    Parameters
    ----------
    data : DataFrame
        Time-series data to apply discrete wavelet transform to. It should be comprised of
        the following two columns:

            - 1st column : Time-stamp, type INTEGER. The stamp values do not need to be in order.
            - 2nd column : Signal values, type DOUBLE or DECIMAL(p,s), must be evenly-spaced in
              time.

    wavelet : str
        Specifies the wavelet filter used for discrete wavelet transform. Valid options include:

            - Daubechies family : 'db1' ~ 'db20'
            - Biorthognal family: 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',
              'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
              'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'
            - Reverse Biorthogal family : 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2',
              'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',
              'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'
            - Coifman family: 'coif1' ~ 'coif5'
            - Symmetric family: 'sym2' ~ 'sym20'

    key : str
        Specifies the time-stamp column in ``data``.

        The column should be of type INTEGER. The values do not need to be in order, but
        must be equal-spaced.

    col : str, optional
        Specifies the signal values for wavelet transformation, should be of type DOUBLE
        or DECIMAL(p,s).

        If not specified, it defaults to the first non-key column of ``data``.

    boundary : str, optional
        Specifies the padding method for boundary values. Valid options include:

            - 'zero' : Zero padding
            - 'symmetric' : Symmetric padding
            - 'periodic' : Periodic padding
            - 'reflect' : Reflect padding
            - 'smooth' : Smooth padding
            - 'constant' : Constant padding

        Defaults to 'zero'.

    level : int, optional
        Specifies the decompose level for wavelet transform.

        Defaults to 1.

    Returns
    -------
    A `DWT` object
        The returned `DWT` object contains all related information for the
        wavelet transformation, including wavelet filter, boundary
        extension type, decomposition level, etc. In particular, it contains
        the wavelet transformation of ``data`` in its `coeff_` attribute.

        For more details, see the `Attributes` section of :class:`DWT`.

    Examples
    --------
    Input time-series data for discrete wavelet transformation:

    >>> data.collect()
        ID    VAL
    0    1  266.0
    1    2  145.9
    2    3  181.3
    3    4  119.3
    4    5  180.3
    5    6  168.5
    6    7  231.8
    7    8  224.5
    8    9  192.8
    9   10  122.9
    10  11  336.5
    11  12  185.9
    12  13  194.3
    13  14  149.5

    2-level decomposition of the input data using 'db2' filter, symmetric padding method:

    >>> dwt = wavedec(data=data, wavelet='db2', boundary='symmetric', level=2)

    `dwt` is a `DWT` object containing the decomposition result in its attribute `coeff_`:

    >>> dwt.coeff_.collect()
        LEVEL  ID  COEFFICIENTS
    0       0   0    451.442328
    1       0   1    405.506091
    2       0   2    350.691644
    3       0   3    412.118406
    4       0   4    362.046517
    5       1   0     73.545930
    6       1   1     26.917407
    7       1   2     19.242329
    8       1   3     24.378527
    9       1   4     21.606776
    10      1   5    139.207493
    11      1   6      5.117513
    12      1   7    -27.434285
    13      2   0     35.520329
    14      2   1    -53.885358
    15      2   2     71.258862
    16      2   3     78.767514
    17      2   4    -70.401833
    """
    dwt = DWT(wavelet, boundary, level)
    data = arg('data', data, DataFrame)
    return dwt.transform(data, key, col)

def waverec(dwt, wavelet=None, boundary=None):
    r"""
    Python wrapper for PAL multi-level inverse discrete wavelet transform.

    Parameters
    ----------
    dwt : DWT
        A `DWT` object containing wavelet coefficients as well as other related information for applying
        the inverse transformation.

    wavelet : str, optional
        Specifies the wavelet filter used for discrete wavelet transform. Valid options include:

            - Daubechies family : 'db1' ~ 'db20'
            - Biorthognal family: 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',
              'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
              'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'
            - Reverse Biorthogal family : 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2',
              'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',
              'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'
            - Coifman family: 'coif1' ~ 'coif5'
            - Symmetric family: 'sym2' ~ 'sym20'

        If not provided, the value of `dwt.wavelet` will be used.

    boundary : str, optional
        Specifies the padding method for boundary values. Valid options include:

            - 'zero' : Zero padding
            - 'symmetric' : Symmetric padding
            - 'periodic' : Periodic padding
            - 'reflect' : Reflect padding
            - 'smooth' : Smooth padding
            - 'constant' : Constant padding

        If not provided, the value of `dwt.boundary` will be used.

    Returns
    -------
    DataFrame
        The reconstructed time-series data from inverse discrete wavelet transform, structured as follows:

            - 1st column : ID, type INTEGER, which reflects the order of time-series.
            - 2nd column : VAL, type DOUBLE, the reconstructed time-series data(signal) from wavelet decomposition coefficients.

    Examples
    --------
    Assume `dwt` is a `DWT` object structured as follows:

    >>> dwt.coeff_.collect()
        LEVEL  ID  COEFFICIENTS
    0       0   0    451.442328
    1       0   1    405.506091
    2       0   2    350.691644
    3       0   3    412.118406
    4       0   4    362.046517
    5       1   0     73.545930
    6       1   1     26.917407
    7       1   2     19.242329
    8       1   3     24.378527
    9       1   4     21.606776
    10      1   5    139.207493
    11      1   6      5.117513
    12      1   7    -27.434285
    13      2   0     35.520329
    14      2   1    -53.885358
    15      2   2     71.258862
    16      2   3     78.767514
    17      2   4    -70.401833

    The original time-series data then can be reconstructed as follows:

    >>> rec = waverec(dwt=dwt)
    >>> rec.collect()
        ID  VALUE
    0    0  266.0
    1    1  145.9
    2    2  181.3
    3    3  119.3
    4    4  180.3
    5    5  168.5
    6    6  231.8
    7    7  224.5
    8    8  192.8
    9    9  122.9
    10  10  336.5
    11  11  185.9
    12  12  194.3
    13  13  149.5
    """
    return dwt.inverse(wavelet, boundary)


def wpdec(data, wavelet, key=None, col=None, boundary=None, level=None, order=None):
    r"""
    Python wrapper for PAL multi-level (discrete) wavelet packet transformation.

    Parameters
    ----------
    data : DataFrame
        Time-series data to apply discrete wavelet transform to. It should be comprised of
        the following two columns:

            - 1st column : Time-stamp, type INTEGER. The stamp values do not need to be in order.
            - 2nd column : Signal values, type DOUBLE or DECIMAL(p,s). The values must be evenly-spaced
              in time.

    wavelet : str
        Specifies the wavelet filter used for discrete wavelet transform. Valid options include:

            - Daubechies family : 'db1' ~ 'db20'
            - Biorthognal family: 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',
              'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
              'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'
            - Reverse Biorthogal family : 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2',
              'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',
              'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'
            - Coifman family: 'coif1' ~ 'coif5'
            - Symmetric family: 'sym2' ~ 'sym20'
    key : str
        Specifies the time-stamp column in ``data``.

        The column should be of type INTEGER. The values do not need to be in order, but
        must be equal-spaced.

    col : str, optional
        Specifies the signal values for wavelet transformation, should be of type DOUBLE
        or DECIMAL(p,s).

        If not specified, it defaults to the first non-key column of ``data``.

    boundary : str, optional
        Specifies the padding method for boundary values. Valid options include:

            - 'zero' : Zero padding
            - 'symmetric' : Symmetric padding
            - 'periodic' : Periodic padding
            - 'reflect' : Reflect padding
            - 'smooth' : Smooth padding
            - 'constant' : Constant padding

        Defaults to 'zero'.

    level : int, optional
        Specifies the decompose level for wavelet transform.

        Defaults to 1.

    order : {'index', 'frequency'}
       Specifies the order of node in the wavelet packet coefficients table.

           - 'index' : ordered by the indices of nodes(ascending).
           - 'frequency' : ordered by the frequencies of nodes, from low to high.

    Returns
    -------
    A `DWT` object
        The returned `DWT` object contains all related information for the
        wavelet packet transformation, including wavelet filter, boundary
        extension type, decomposition level, etc. In particular, it contains
        the wavelet packet transformation of ``data`` in its `coeff_` attribute.

        For more details, see the `Attributes` section of :class:`DWT`.

    Examples
    --------
    Input time-series data for discrete wavelet transformation:

    >>> data.collect()
        ID    VAL
    0    1  266.0
    1    2  145.9
    2    3  181.3
    3    4  119.3
    4    5  180.3
    5    6  168.5
    6    7  231.8
    7    8  224.5
    8    9  192.8
    9   10  122.9
    10  11  336.5
    11  12  185.9
    12  13  194.3
    13  14  149.5

    2-level wavelet packet transformation of the input data using 'db2' filter and symmetric padding method,
    with wavelet packet coefficients ordered by nodes' frequencies.

    >>> wpres = wpdec(data=data, wavelet='db2', boundary='symmetric', level=2)

    `wpres` is a `DWT` object containing the decomposition result in its attribute `coeff_`:

    >>> wpres.coeff_.collect()
        NODE  ID  COEFFICIENTS
    0      0   0    451.442328
    1      0   1    405.506091
    2      0   2    350.691644
    3      0   3    412.118406
    4      0   4    362.046517
    5      1   0     35.520329
    6      1   1    -53.885358
    7      1   2     71.258862
    8      1   3     78.767514
    9      1   4    -70.401833
    10     3   0     28.554022
    11     3   1    -11.228318
    12     3   2    -57.112074
    13     3   3    -16.468003
    14     3   4    -19.933824
    15     2   0     87.523979
    16     2   1     59.195043
    17     2   2     16.514617
    18     2   3    131.581926
    19     2   4    -27.289140
    >>> wpres.stats_.collect()
                   NAME                     VAL
    0  LEVEL_COEFF_SIZE  {"coeffSize":[14,8,5]}
    >> wpres.packet
    True
    """
    dwt = DWT(wavelet, boundary, level, True, order)
    data = arg('data', data, DataFrame)
    return dwt.transform(data, key, col)

def wprec(dwt, wavelet=None, boundary=None):
    r"""
    Python wrapper for PAL multi-level inverse discrete wavelet transform.

    Parameters
    ----------
    dwt : DWT
        A DWT object containing wavelet packet coefficients as well as other related information to apply
        the inverse transformation(i.e. reconstruction).

    wavelet : str, optional
        Specifies the wavelet filter used for discrete wavelet transform. Valid options include:

            - Daubechies family : 'db1' ~ 'db20'
            - Biorthognal family: 'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',
              'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5',
              'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8'
            - Reverse Biorthogal family : 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2',
              'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5',
              'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8'
            - Coifman family: 'coif1' ~ 'coif5'
            - Symmetric family: 'sym2' ~ 'sym20'

        If not specified, the value in `dwt.wavelet` will be used.

    boundary : str, optional
        Specifies the padding method for boundary values. Valid options include:

            - 'zero' : Zero padding
            - 'symmetric' : Symmetric padding
            - 'periodic' : Periodic padding
            - 'reflect' : Reflect padding
            - 'smooth' : Smooth padding
            - 'constant' : Constant padding

        If not specified, the value in `dwt.boundary` will be used.

    Returns
    -------
    DataFrame
        The reconstructed time-series data from inverse wavelet packet transform, structured as follows:

            - 1st column : ID, type INTEGER, which reflects the order of time-series.
            - 2nd column : VALUE, type DOUBLE, the reconstructed time-series data(signal) from wavelet decomposition coefficients.

    Examples
    --------
    Assume `dwt` is a `DTW` object with the following attributes:

    >>> dwt.coeff_.collect()
        NODE  ID  COEFFICIENTS
    0      0   0    451.442328
    1      0   1    405.506091
    2      0   2    350.691644
    3      0   3    412.118406
    4      0   4    362.046517
    5      1   0     35.520329
    6      1   1    -53.885358
    7      1   2     71.258862
    8      1   3     78.767514
    9      1   4    -70.401833
    10     3   0     28.554022
    11     3   1    -11.228318
    12     3   2    -57.112074
    13     3   3    -16.468003
    14     3   4    -19.933824
    15     2   0     87.523979
    16     2   1     59.195043
    17     2   2     16.514617
    18     2   3    131.581926
    19     2   4    -27.289140
    >>> dwt.stats_.collect()
                   NAME                     VAL
    0  LEVEL_COEFF_SIZE  {"coeffSize":[14,8,5]}
    >>> dwt.packet
    True

    The original time-series data then can be reconstructed as follows:

    >>> rec = wprec(dwt)
    >>> rec.collect()
        ID  VALUE
    0    0  266.0
    1    1  145.9
    2    2  181.3
    3    3  119.3
    4    4  180.3
    5    5  168.5
    6    6  231.8
    7    7  224.5
    8    8  192.8
    9    9  122.9
    10  10  336.5
    11  11  185.9
    12  12  194.3
    13  13  149.5
    """
    return dwt.inverse(wavelet, boundary)
