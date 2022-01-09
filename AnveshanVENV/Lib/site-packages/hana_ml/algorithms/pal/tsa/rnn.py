"""
This module contains Python wrappers of recurrent neural networks(RNN)
for time-series analysis in PAL.
In particular, GRU with attention mechanism.

The following class are available:

    * :class:`GRUAttention`
"""

#pylint: disable=too-many-lines, line-too-long, too-many-locals
#pylint: disable=consider-using-f-string
import logging
import uuid
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    pal_param_register,
    try_drop,
    require_pal_usable,
    ListOfStrings
)
from hana_ml.algorithms.pal.sqlgen import trace_sql
from .lstm import LSTM#pylint:disable=unused-import

logger = logging.getLogger(__name__)#pylint: disable=invalid-name

class GRUAttention(PALBase):#pylint: disable=too-many-instance-attributes, too-few-public-methods
    r"""
    Gated Recurrent Units(GRU) based encoder-decoder model with `Attention` mechanism for time series prediction.

    Parameters
    ----------
    learning_rate : float, optional
        Learning rate for gradient descent.

        Defaults to 0.05.
    batch_size : int, optional
        Number of pieces of data for training in one iteration.

        Defaults to 32.
    time_dim : int, optional
        It specifies how many time steps in a sequence that will be trained by LSTM/GRU and then for time series prediction.

        The value of it must be smaller than the length of input time series minus 1.

        Defaults to 16.
    hidden_dim : int, optional
        Number of hidden neurons within every GRU layer.

        Defaults to 64.
    num_layers : int, optional
        Number of layers in GRU unit at encoder part and decoder part.

        Defaults to 1.
    max_iter : int, optional
        Number of batches of data by which attention model is trained.

        Defaults to 1000.
    interval : int, optional
        Output the average loss within every ``interval`` iterations.

        Defaults to 100.

    Attributes
    ----------
    loss_ : DateFrame

        LOSS.

    model_ : DataFrame

        Model content.

    Examples
    --------
    Input dataframe df:

    >>> df.head(3).collect()
       TIMESTAMP  SERIES
    0          0    20.7
    1          1    17.9
    2          2    18.8

    Create a LSTM model:

    >>> from hana_ml.algorithms.pal.tsa import rnn
    >>> att = rnn.GRUAttention(max_iter=1000,
                                learning_rate=0.01,
                                batch_size=32,
                                hidden_dim=128,
                                num_layers=1,
                                interval=1)

    Perform fit on the given data:

    >>> att.fit(self.df)

    Peform predict on the fittd model:

    >>> res = att.predict(self.df_predict)

    Expected output:

    >>> res.head(3).collect()
       ID                                      MODEL_CONTENT
    0   0  {"PreprocessArgs1":[16.191999999999998],"Prepr...
    1   1  2943487073086303,-0.14125998608870073,0.016476...
    2   2  9670601,0.26983407416799118,-0.125378179522916...

    """

    def __init__(self,#pylint: disable=too-many-arguments
                 learning_rate=None,
                 batch_size=None,
                 time_dim=None,
                 hidden_dim=None,
                 num_layers=None,
                 max_iter=None,
                 interval=None):

        setattr(self, 'hanaml_parameters', pal_param_register())
        super(GRUAttention, self).__init__()
        self.learning_rate = self._arg('learning_rate', learning_rate, (float, int))
        self.batch_size = self._arg('batch_size', batch_size, int)
        self.time_dim = self._arg('time_dim', time_dim, int)
        self.hidden_dim = self._arg('hidden_dim', hidden_dim, int)
        self.num_layers = self._arg('num_layers', num_layers, int)
        self.max_iter = self._arg('max_iter', max_iter, int)
        self.interval = self._arg('interval', interval, int)

    @trace_sql
    def fit(self, data, key=None, endog=None, exog=None):#pylint: disable=too-many-arguments,too-many-branches, too-many-statements
        """
        Generates a `GRUAttention` model with given data and specified parameters.

        Parameters
        ----------
        data : DataFrame
            Input data, should contain at least 2 columns described as follows:

              - Index(i.e. time-stamp) column, type INTEGER. The time-stamps do not need to be in order,
                but must be unique and evenly spaced.
              - Time-series values column, type INTEGER, DOUBLE or DECIMAL(p,s).

        key : str, optional
            Specifies the name of the `index` column in ``data``.

            If not provided, it defaults to:

              - 1st column of ``data`` if ``data`` is not indexed or indexed by multiple columns.
              - the index column of ``data`` if ``data`` is indexed by a single column.

        endog : str, optional
            Specifies the name of endogenous variable, i.e. time series values in ``data``.
            The type of ``endog`` column could be INTEGER, DOUBLE or DECIMAL(p,s).

            Defaults to the 1st non-key column of ``data`` if not provided.

        exog : str or ListOfStrings, optional
            Specifies the name of exogenous variable. The type of an ``exog`` column
            could be INTEGER, DOUBLE, DECIMAL(p,s), VARCHAR or NVARCHAR.

            Defaults to all columns in ``data`` except the ``key`` column and
            the ``endog`` column.

        Returns
        -------
        A fitted object of class "GRUAttention".
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        conn = data.connection_context
        require_pal_usable(conn)
        cols = data.columns
        key = self._arg('key', key, str)
        if key is not None:
            key = self._arg('key', key.lower(), {col.lower():col for col in cols})
        elif isinstance(data.index, str):
            key = data.index
        else:
            key = cols[0]
        cols.remove(key)
        endog = self._arg('endog', endog, str)
        if endog is not None:
            endog = self._arg('endog', endog.lower(), {col.lower():col for col in cols})
        else:
            endog = cols[0]
        cols.remove(endog)
        if isinstance(exog, str):
            exog = [exog]
        exog = self._arg('exog', exog, ListOfStrings)
        if exog is None:
            exog = cols
        data_ = data[[key, endog] + exog]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        loss_tbl = '#PAL_ATTENTION_LOSS_TBL_{}_{}'.format(self.id, unique_id)
        model_tbl = '#PAL_ATTENTION_MODEL_TBL_{}_{}'.format(self.id, unique_id)
        outputs = [loss_tbl, model_tbl]
        param_rows = [
            ('LEARNING_RATE', None, self.learning_rate, None),
            ('BATCH_SIZE', self.batch_size, None, None),
            ('TIME_DIM', self.time_dim, None, None),
            ('HIDDEN_DIM', self.hidden_dim, None, None),
            ('NUM_LAYERS', self.num_layers, None, None),
            ('MAX_ITER', self.max_iter, None, None),
            ('INTERVAL', self.interval, None, None)]
        try:
            self._call_pal_auto(conn,
                                'PAL_ATTENTION_TRAIN',
                                data_,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            try_drop(conn, outputs)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            try_drop(conn, outputs)
            raise
        #pylint: disable=attribute-defined-outside-init
        self.loss_ = conn.table(loss_tbl)
        self.model_ = conn.table(model_tbl)
        return self

    @trace_sql
    def predict(self, data, top_k_attributions=None):
        """
        Makes time series forecast based on the trained `GRUAttention` model.

        Parameters
        ----------

        data : DataFrame
            Data for prediction, structured as follows:
                - 1st column : Record IDs.
                - Other columsn: Columns for holding the time-series and external data values
                  for all records, arranged in time-order. The columns' data type need to be
                  consistent with that of time-series or external data.

            .. Note::
                In this DataFrame, eahc row contains a piece of time-series data with with external data
                for prediction, and the number of columns for time-series values should be equal to
                ``time_dim`` * (**M**-1), where **M** is the number of columns of the input data
                in the training phase.

        top_k_attributions : int, optional
            Specifies the number of features with highest attributions to output.

            This value needs to be smaller than the length of time series data for prediction.

            Defaults to 10.

        Returns
        -------
        DataFrame
            The aggerated forecasted values.
            Forecasted values, structured as follows:

              - ID, type INTEGER, representing record ID.
              - VALUE, type DOUBLE, containing the forecast value of the corresponding record.
              - REASON_CODE, type NCLOB, containing sorted SHAP values for test data at each time step.
        """
        if getattr(self, 'model_', None) is None:
            msg = ('Model not initialized. Perform a fit first.')
            logger.error(msg)
            raise FitIncompleteError(msg)
        conn = data.connection_context
        require_pal_usable(conn)
        param_rows = [
            ("TOP_K_ATTRIBUTIONS", top_k_attributions, None, None)]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_ATTENTION_PREDICT_RESULT_TBL_{}_{}".format(self.id, unique_id)
        try:
            self._call_pal_auto(conn,
                                'PAL_ATTENTION_PREDICT',
                                data,
                                self.model_,
                                ParameterTable().with_data(param_rows),
                                result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, result_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, result_tbl)
            raise
        result_df = conn.table(result_tbl)
        return result_df
