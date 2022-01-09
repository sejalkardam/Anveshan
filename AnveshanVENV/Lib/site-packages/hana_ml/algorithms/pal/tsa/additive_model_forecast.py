"""
This module contains Python wrapper for PAL Addtive Model Forecast algorithm.

The following class are available:

    * :class:`AdditiveModelForecast`
"""
#pylint: disable=too-many-instance-attributes, too-few-public-methods
#pylint: disable=too-many-lines, line-too-long, invalid-name, too-many-branches
#pylint: disable=too-many-arguments, too-many-locals, attribute-defined-outside-init
#pylint: disable=super-with-arguments, c-extension-no-member
#pylint: disable=consider-using-f-string
import logging
import uuid
import warnings
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.algorithms.pal.sqlgen import trace_sql
from hana_ml.algorithms.pal.utility import check_pal_function_exist
from hana_ml.algorithms.pal.pal_base import (
    PALBase,
    ParameterTable,
    ListOfStrings,
    pal_param_register,
    try_drop,
    require_pal_usable,
    execute_logged
)
logger = logging.getLogger(__name__)

class _AdditiveModelForecastBase(PALBase):
    """
    Additive model forecast base class.
    """
    seasonality_map = {'auto': -1, 'false': 0, 'true': 1}
    def __init__(self,
                 growth=None,
                 logistic_growth_capacity=None,
                 seasonality_mode=None,
                 seasonality=None,
                 num_changepoints=None,
                 changepoint_range=None,
                 regressor=None,
                 changepoints=None,
                 yearly_seasonality=None,
                 weekly_seasonality=None,
                 daily_seasonality=None,
                 seasonality_prior_scale=None,
                 holiday_prior_scale=None,
                 changepoint_prior_scale=None):
        super(_AdditiveModelForecastBase, self).__init__()
        setattr(self, 'hanaml_parameters', pal_param_register())
        #P, D, Q in PAL has been combined to be one parameter `order`
        self.growth = self._arg('growth', growth, str)
        self.logistic_growth_capacity = self._arg('logistic_growth_capacity', logistic_growth_capacity, float)
        if self.growth == 'logistic' and self.logistic_growth_capacity is None:
            msg = "`logistic_growth_capacity` is mandatory when `growth` is 'logistic'."
            logger.error(msg)
            raise ValueError(msg)

        self.seasonality_mode = self._arg('seasonality_mode', seasonality_mode, str)

        self.num_changepoints = self._arg('num_changepoints', num_changepoints, int)
        self.changepoint_range = self._arg('changepoint_range', changepoint_range, float)

        self.changepoints = self._arg('changepoints', changepoints, ListOfStrings)
        self.yearly_seasonality = self._arg('yearly_seasonality', yearly_seasonality, self.seasonality_map)
        self.weekly_seasonality = self._arg('weekly_seasonality', weekly_seasonality, self.seasonality_map)
        self.daily_seasonality = self._arg('daily_seasonality', daily_seasonality, self.seasonality_map)
        self.seasonality_prior_scale = self._arg('seasonality_prior_scale', seasonality_prior_scale, float)
        self.holidays_prior_scale = self._arg('holiday_prior_scale', holiday_prior_scale, float)
        self.changepoint_prior_scale = self._arg('changepoint_prior_scale', changepoint_prior_scale, float)

        if isinstance(regressor, str):
            regressor = [regressor]
        self.regressor = self._arg('regressor', regressor, ListOfStrings)

        if isinstance(seasonality, str):
            seasonality = [seasonality]
        self.seasonality = self._arg('seasonality', seasonality, ListOfStrings)

    @trace_sql
    def _fit(self, data, holiday, categorical_variable):
        """
        Additive model forecast fit function.
        """
        conn = data.connection_context
        require_pal_usable(conn)

        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable,
                                         ListOfStrings)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        holiday_tbl = None
        if holiday is None:
            holiday_tbl = "#PAL_ADDITIVE_MODEL_FORECAST_HOLIDAY_TBL_{}_{}".format(self.id, unique_id)
            with conn.connection.cursor() as cur:
                execute_logged(cur,
                               'CREATE LOCAL TEMPORARY COLUMN TABLE {} ("ts" TIMESTAMP, "NAME" VARCHAR(255),\
                               "LOWER_WINDOW" INTEGER, "UPPER_WINDOW" INTEGER)'.format(holiday_tbl),
                               conn.sql_tracer,
                               conn)
                holiday = conn.table(holiday_tbl)
            if not conn.connection.getautocommit():
                conn.connection.commit()
        model_tbl = '#PAL_ADDITIVE_MODEL_FORECAST_MODEL_TBL_{}_{}'.format(self.id, unique_id)
        outputs = [model_tbl]
        param_rows = [
            ('GROWTH', None, None, self.growth),
            ('CAP', None, self.logistic_growth_capacity, None),
            ('SEASONALITY_MODE', None, None, self.seasonality_mode),
            ('NUM_CHANGEPOINTS', self.num_changepoints, None, None),
            ('CHANGEPOINT_RANGE', None, self.changepoint_range, None),
            ('YEARLY_SEASONALITY', self.yearly_seasonality, None, None),
            ('WEEKLY_SEASONALITY', self.weekly_seasonality, None, None),
            ('DAILY_SEASONALITY', self.daily_seasonality, None, None),
            ('SEASONALITY_PRIOR_SCALE', None, self.seasonality_prior_scale, None),
            ('HOLIDAYS_PRIOR_SCALE', None, self.holidays_prior_scale, None),
            ('CHANGEPOINT_PRIOR_SCALE', None, self.changepoint_prior_scale, None)
            ]

        if self.changepoints is not None:
            for changepoint in self.changepoints:
                param_rows.extend([('CHANGE_POINT', None, None, changepoint)])

        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, variable)
                               for variable in categorical_variable])

        if self.seasonality is not None:
            for s in self.seasonality:
                param_rows.extend([('SEASONALITY', None, None, s)])

        if self.regressor is not None:
            for r in self.regressor:
                param_rows.extend([('REGRESSOR', None, None, r)])

        try:
            self._call_pal_auto(conn,
                                'PAL_ADDITIVE_MODEL_ANALYSIS',
                                data,
                                holiday,
                                ParameterTable().with_data(param_rows),
                                *outputs)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            if holiday_tbl is not None:
                try_drop(conn, holiday_tbl)
            try_drop(conn, outputs)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            if holiday_tbl is not None:
                try_drop(conn, holiday_tbl)
            try_drop(conn, outputs)
            raise

        self.model_ = conn.table(model_tbl)

    @trace_sql
    def _predict(self,
                 data,
                 logistic_growth_capacity=None,
                 interval_width=None,
                 uncertainty_samples=None,
                 show_explainer=None,
                 decompose_seasonality=None,
                 decompose_holiday=None):

        conn = data.connection_context
        require_pal_usable(conn)

        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        data_tbl = None
        result_tbl = "#PAL_ADDITIVE_MODEL_FORECAST_RESULT_TBL_{}_{}".format(self.id, unique_id)
        decompose_tbl = "#PAL_ADDITIVE_MODEL_FORECAST_DECOMPOSITION_TBL_{}_{}".format(self.id, unique_id)

        if show_explainer is not True:
            param_rows = [
                ("CAP", None, logistic_growth_capacity, None),
                ("INTERVAL_WIDTH", None, interval_width, None),
                ("UNCERTAINTY_SAMPLES", uncertainty_samples, None, None)]
        else:
            param_rows = [
                ("CAP", None, logistic_growth_capacity, None),
                ("INTERVAL_WIDTH", None, interval_width, None),
                ("UNCERTAINTY_SAMPLES", uncertainty_samples, None, None),
                ("EXPLAIN_SEASONALITY", decompose_seasonality, None, None),
                ("EXPLAIN_HOLIDAY", decompose_holiday, None, None)]
        try:
            if show_explainer is not True:
                self._call_pal_auto(conn,
                                    'PAL_ADDITIVE_MODEL_PREDICT',
                                    data,
                                    self.model_,
                                    ParameterTable().with_data(param_rows),
                                    result_tbl)
            else:
                if check_pal_function_exist(conn, 'ADDITIVE_MODEL_EXPLAIN%', like=True):
                    self._call_pal_auto(conn,
                                        'PAL_ADDITIVE_MODEL_EXPLAIN',
                                        data,
                                        self.model_,
                                        ParameterTable().with_data(param_rows),
                                        result_tbl,
                                        decompose_tbl)
                else:
                    msg = 'The version of SAP HANA does not support additive_model_forecast explainer. Please set show_explainer=False!'
                    logger.error(msg)
                    raise ValueError(msg)
        except dbapi.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err))
            if data_tbl is not None:
                try_drop(conn, data_tbl)
            try_drop(conn, result_tbl)
            raise
        except pyodbc.Error as db_err:
            msg = str(conn.hana_version())
            logger.exception("HANA version: %s. %s", msg, str(db_err.args[1]))
            if data_tbl is not None:
                try_drop(conn, data_tbl)
            try_drop(conn, result_tbl)
            raise
        if show_explainer is True:
            self.explainer_ = conn.table(decompose_tbl)

        return conn.table(result_tbl)

class AdditiveModelForecast(_AdditiveModelForecastBase):
    r"""
    Additive Model Forecast use a decomposable time series model with three main components: trend, seasonality, and holidays or event.

    Note that this function is a new function in SAP HANA SPS05 and Cloud.

    Parameters
    ----------

    growth : {'linear', 'logistic'}, optional

        Specify a trend, which could be either linear or logistic.

        Defaults to 'linear'.

    logistic_growth_capacity: float, optional

        Specify the carrying capacity for logistic growth.
        Mandatory and valid only when ``growth`` is 'logistic'.

        No default value.
    seasonality_mode : {'additive', 'multiplicative'}, optional

        Mode for seasonality, either additive or muliplicative.

        Defaults to 'additive'.
    seasonality : str or list of str, optional

        Add seasonality to model, is a json format, include:

          - NAME
          - PERIOD
          - FOURIER_ORDER
          - PRIOR_SCALE, optional
          - MODE, optional

        Each str is json format such as '{ "NAME": "MONTHLY", "PERIOD":30, "FOURIER_ORDER":5 }'.
        A seasonality parameter must include NAME, PERIOD, and FOURIER_ORDER elements.
        PRIOR_SCALE and MODE elements are optional.
        FOURIER_ORDER determines how quickly the seasonality can change.
        PRIOR_SCALE controls the amount of regularization.
        No seasonality will be added to the model if this parameter is not provided.

        No default value.

    num_changepoints : int, optional

        Number of potential changepoints.
        Not effective if ``changepoints`` is provided.

        Defaults to 25 if not provided.

    changepoint_range : float, optional

        Proportion of history in which trend changepoints will be estimated.
        Not effective if ``changepoints`` is provided.

        Defaults to 0.8.

    regressor : str, optional

        Specify the regressor, include:
          - NAME
          - PRIOR_SCALE
          - STANDARDIZE
          - MODE: "additive" or 'multiplicative'.

        Each str is json format such as '{ "NAME": "X1", "PRIOR_SCALE":4, "MODE": "additive" }'.
        PRIOR_SCALE controls for the amount of regularization;
        STANDARDIZE specifies whether or not the regressor is standardized.
        No default value.

    changepoints : list of str, optional,

        Specify a list of changepoints in the format of timestamp,
        such as ['2019-01-01 00:00:00, '2019-02-04 00:00:00']

        No default value.

    yearly_seasonality : {'auto', 'false', 'true'}, optional

        Specify whether or not to fit yearly seasonality.

        'false' and 'true' simply corresponds to their logical meaning, while 'auto' means automatically determined from the input data.

        Defaults to 'auto'.

    weekly_seasonality : {'auto', 'false', 'true'}, optional

        Specify whether or not to fit the weekly seasonality.

        'auto' means automatically determined from input data.

        Defaults to 'auto'.

    daily_seasonality : {'auto', 'false', 'true'}, optional

        Specify whether or not to fit the daily seasonality.

        'auto' means automatically determined from input data.

        Defaults to 'auto'.

    seasonality_prior_scale : float, optional

        Parameter modulating the strength of the seasonality model.

        Defaults to 10 if not provided.

    holiday_prior_scale : float, optional

        Parameter modulating the strength of the holiday components model.

        Defaults to 10 if not provided.

    changepoint_prior_scale : float, optional

        Parameter modulating the flexibility of the automatic changepoint selection.

        Defaults to 0.05 if not provided.

    categorical_variable : str or list of str, optional
        Indicate which variables are treated as categorical. The default behavior is:

        - string: categorical.
        - integer and double: numerical.

        Detected from input data and valid only for integer or string column.

    Attributes
    ----------

    model_ : DataFrame

        Model content.

    explainer_ : DataFrame

        The decomposition of trend, seasonal, holiday and exogenous variables.
        Only contains value when ``show_explainer=True`` in the predict function.


    Examples
    --------

    Input dataframe df_fit:

    >>> df_fit.head(5).collect()
               ts         y
    0  2007-12-10  9.590761
    1  2007-12-11  8.519590
    2  2007-12-12  8.183677
    3  2007-12-13  8.072467
    4  2007-12-14  7.893572

    Create an Additive Model Forecast model:

    >>> amf = additive_model_forecast.AdditiveModelForecast(growth='linear')

    Perform fit on the given data:

    >>> amf.fit(data=df_fit)

    Expected output:

    >>> amf.model_.collect()
       ROW_INDEX                                      MODEL_CONTENT
    0          0  {"GROWTH":"linear","FLOOR":0.0,"SEASONALITY_MO...

    Perform predict on the model:

    Input dataframe df_predict for prediction:

    >>> df_predict.head(5).collect()
               ts    y
    0  2008-03-09  0.0
    1  2008-03-10  0.0
    2  2008-03-11  0.0
    3  2008-03-12  0.0
    4  2008-03-13  0.0

    >>> result = amf.predict(data=df2)

    Expected output:

    >>> result.collect()
               ts      YHAT  YHAT_LOWER  YHAT_UPPER
    0  2008-03-09  7.676880    6.930349    8.461546
    1  2008-03-10  8.147574    7.387315    8.969112
    2  2008-03-11  7.410452    6.630115    8.195562
    3  2008-03-12  7.198807    6.412776    7.977391
    4  2008-03-13  7.087702    6.310826    7.837083

    If you want to see the decomposed result of predict result, you could set show_explainer = True:

    >>> result = amf.predict(df_predict,
                             show_explainer=True,
                             decompose_seasonality=False,
                             decompose_holiday=False)

    Show the ``explainer_``:

    >>> amf.explainer_.head(5).collect()
                ts     TREND                                SEASONAL HOLIDAY EXOGENOUS
    0   2008-03-09  7.432172   {"seasonalities":0.24470822257259804}      {}        {}
    1   2008-03-10  7.390030     {"seasonalities":0.757544365973254}      {}        {}
    2   2008-03-11  7.347887   {"seasonalities":0.06256440574150749}      {}        {}
    3   2008-03-12  7.305745  {"seasonalities":-0.10693834906369426}      {}        {}
    4   2008-03-13  7.263603  {"seasonalities":-0.17590059499681369}      {}        {}

    """
    def fit(self, data, key=None, endog=None, exog=None, holiday=None, categorical_variable=None):
        """
        Additive model forecast fit function.

        Parameters
        ----------

        data : DataFrame

            Input data. The structure is as follows.

            - The first column: index (ID), type TIMESTAMP, SECONDDATE or DATE.
            - The second column: raw data, type INTEGER or DECIMAL(p,s).
            - Other columns: external data, type INTEGER, DOUBLE or DECIMAL(p,s).

        key : str, optional

            The timestamp column of data. The type of key column is TIMESTAMP, SECONDDATE, or DATE.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        endog : str, optional

            The endogenous variable, i.e. time series. The type of endog column is INTEGER, DOUBLE, or DECIMAL(p, s).

            Defaults to the first non-key column of data if not provided.

        exog : str or a list of str, optional

            An optional array of exogenous variables. The type of exog column is INTEGER, DOUBLE, or DECIMAL(p, s).

            Defaults to None. Please set this parameter explicitly if you have exogenous variables.

        holiday : DataFrame

            Input holiday data. The structure is as follows.

            - The first column : index, timestamp
            - The second column : name, varchar
            - The third column : lower window of holiday, int, optional
            - The last column : upper window of holiday, int, optional

            Defaults to None.

        categorical_variable : str or ist of str, optional

            Specifies INTEGER columns specified that should be be treated as categorical.

            Other INTEGER columns will be treated as continuous.

        Returns
        -------
        A fitted object of class "AdditiveModelForecast".
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
        if data is None:
            msg = ('The data for fit cannot be None!')
            logger.error(msg)
            raise ValueError(msg)

        # validate key, endog, exog
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

        if key is not None and key not in cols:
            msg = ('Please select key from name of columns!')
            logger.error(msg)
            raise ValueError(msg)
        cols.remove(key)

        endog = self._arg('endog', endog, str)
        if endog is not None:
            if endog not in cols:
                msg = ('Please select endog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            endog = cols[0]
        cols.remove(endog)

        if exog is not None:
            if isinstance(exog, str):
                exog = [exog]
            exog = self._arg('exog', exog, ListOfStrings)
            if set(exog).issubset(set(cols)) is False:
                msg = ('Please select exog from name of columns!')
                logger.error(msg)
                raise ValueError(msg)
        else:
            exog = []

        data_ = data[[key] + [endog] + exog]
        self.fit_data = data_

        super(AdditiveModelForecast, self)._fit(data_, holiday, categorical_variable)
        return self

    def predict(self,
                data,
                key=None,
                logistic_growth_capacity=None,
                interval_width=None,
                uncertainty_samples=None,
                show_explainer=None,
                decompose_seasonality=None,
                decompose_holiday=None):
        """
        Makes time series forecast based on the estimated Additive Model Forecast model.

        Parameters
        ----------

        data : DataFrame, optional

            Index and exogenous variables for forecast. \
            The structure is as follows.

            - First column: Index (ID), type TIMESTAMP, SECONDDATE or DATE.
            - Second column: Placeholder column for forecast values, type DOUBLE or DECIMAL(p,s).
            - Other columns : external data, type INTEGER, DOUBLE or DECIMAL(p,s).

        key : str, optional

            The timestamp column of data. The data type of key column should be
            TIMESTAMP, DATE or SECONDDATE.

            Defaults to the first column of data if the index column of data is not provided.
            Otherwise, defaults to the index column of data.

        logistic_growth_capacity: float, optional

            specify the carrying capacity for logistic growth.
            Mandatory and valid only when ``growth`` is 'logistic'.

            Defaults to None.
        interval_width : float, optional

            Width of the uncertainty intervals.

            Defaults to 0.8.

        uncertainty_samples : int, optional

            Number of simulated draws used to estimate uncertainty intervals.

            Defaults to 1000.

        show_explainer : bool, optional
            Indicate whether to invoke the AdditiveModelForecast with explainations function in the predict.
            If true, the contributions of trend, seasonal, holiday and exogenous variables are
            shown in a attribute called ``explainer_`` of the AdditiveModelForecast instance.

            Defaults to False.

        decompose_seasonality : bool, optional
            Specify whether or not seasonal component will be decomposed.
            Valid only when ``show_explainer`` is True.

            Defaults to False.

        decompose_holiday : bool, optional
            Specify whether or not holiday component will be decomposed.
            Valid only when ``show_explainer`` is True.

            Defaults to False.

        Returns
        -------

        DataFrame
            Forecasted values, structured as follows:

              - ID, type timestamp.
              - YHAT, type DOUBLE, forecast value.
              - YHAT_LOWER, type DOUBLE, lower bound of confidence region.
              - YHAT_UPPER, type DOUBLE, higher bound of confidence region.
        """
        if getattr(self, 'model_', None) is None:
            raise FitIncompleteError("Model not initialized. Perform a fit first.")

        logistic_growth_capacity = self._arg('logistic_growth_capacity', logistic_growth_capacity, float)
        interval_width = self._arg('interval_width', interval_width, float)
        uncertainty_samples = self._arg('uncertainty_samples', uncertainty_samples, int)
        show_explainer = self._arg('show_explainer', show_explainer, bool)

        if show_explainer is True:
            decompose_seasonality = self._arg('decompose_seasonality', decompose_seasonality, bool)
            decompose_holiday = self._arg('decompose_holiday', decompose_holiday, bool)

        index = data.index
        cols = data.columns
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
        exog = cols
        data_ = data[[key] + exog]
        self.predict_data = data_

        return super(AdditiveModelForecast, self)._predict(data_,
                                                           logistic_growth_capacity,
                                                           interval_width,
                                                           uncertainty_samples,
                                                           show_explainer,
                                                           decompose_seasonality,
                                                           decompose_holiday)
