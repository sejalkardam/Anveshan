"""
This module contains Python wrapper for SAP HANA PAL unified-classification.

The following classes are available:
    * :class:`UnifiedClassification`
"""

#pylint: disable=too-many-lines
#pylint: disable=line-too-long
#pylint: disable=too-many-locals
#pylint: disable=too-many-arguments
#pylint: disable=ungrouped-imports
#pylint: disable=relative-beyond-top-level
#pylint: disable=no-member
#pylint: disable=consider-using-f-string, consider-iterating-dictionary
import logging
import uuid
import pandas as pd
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.visualizers.model_report import (
    _UnifiedClassificationReportBuilder
)
from hana_ml.ml_base import try_drop
from hana_ml.ml_exceptions import FitIncompleteError
from hana_ml.dataframe import quotename
from hana_ml.algorithms.pal.utility import AMDPHelper
from .sqlgen import trace_sql
from .pal_base import (
    PALBase,
    ParameterTable,
    require_pal_usable,
    pal_param_register,
    ListOfStrings,
    ListOfTuples
)



logger = logging.getLogger(__name__) #pylint: disable=invalid-name

def json2tab_for_reason_code(data):
    """
    Transform json formated reason code to table formated one.

    parameters
    ----------
    data : DataFrame
        DataFrame contains the reason code.

    """
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    temp_tab_name = "#REASON_CODE_" + unique_id
    data.connection_context.sql("""
        SELECT '{' || '"ID":' || ID || ',"REASON_CODE":' || REASON_CODE || '}' AS REASON_CODE
        FROM """ + "{}.{}".format(quotename(data.source_table['SCHEMA_NAME']),
                                  quotename(data.source_table['TABLE_NAME']))).save(temp_tab_name, force=True)
    return data.connection_context.sql("""
        SELECT JT.*
        FROM JSON_TABLE({}.REASON_CODE, '$'
        COLUMNS
            (
                ID INT PATH '$.ID',
                NESTED PATH '$.REASON_CODE[*]'
                COLUMNS
                    (
                        "attr" VARCHAR(255) PATH '$.attr',
                        "pct" DOUBLE PATH '$.pct',
                        "val" DOUBLE PATH '$.val'
                    )
            )
            ) AS JT""".format(temp_tab_name))

class UnifiedClassification(PALBase, _UnifiedClassificationReportBuilder, AMDPHelper):#pylint: disable=too-many-instance-attributes
    """
    The Python wrapper for SAP HANA PAL unified-classification function.

    Compared with the original classification interfaces,
    new features supported are listed below:

        - Classification algorithms easily switch
        - Dataset automatic partition
        - Model evaluation procedure provided
        - More metrics supported

    Parameters
    ----------

    func : str

        The name of a specified classification algorithm.
        The following algorithms are supported:

            - 'DecisionTree'
            - 'HybridGradientBoostingTree'
            - 'LogisticRegression'
            - 'MLP'
            - 'NaiveBayes'
            - 'RandomDecisionTree'
            - 'SVM'

        .. Note ::
            'LogisticRegression' contains both binary-class logistic-regression as well as multi-class logistic-regression functionalities. \
            By default the functionality is assumed to be binary-class. If you want to shift to multi-class logistic-regression, \
            please set ``func`` to be 'LogisticRegression' and specify ``multi-class`` = True in ``**kwargs``.

    **kwargs : keyword arguments

        Arbitrary keyword arguments and please referred to the responding algorithm for the parameters' key-value pair.

        **Note that some parameters are disabled in the classification algorithm!**

            - **'DecisionTree'** : :class:`~hana_ml.algorithms.pal.trees.DecisionTreeClassifier`

              - Disabled parameters: output_rules, output_confusion_matrix
              - Parameters removed from initialization but can be specified in fit(): categorical_variable, bins, priors

            - **'HybridGradientBoostingTree'** : :class:`~hana_ml.algorithms.pal.trees.HybridGradientBoostingClassifier`

              - Disabled parameters: calculate_importance, calculate_cm
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'LogisticRegression'** :class:`~hana_ml.algorithms.pal.linear_model.LogisticRegression`

              - Disabled parameters: pmml_export
              - Parameters removed from initialization but can be specified in fit(): categorical_variable, class_map0, class_map1

            - **'MLP'** : :class:`~hana_ml.algorithms.pal.neural_network.MLPClassifier`

              - Disabled parameters: functionality
              - Parameters removed from initialization but can be specified in fit(): categorical_variable

            - **'NaiveBayes'** : :class:`~hana_ml.algorithms.pal.naive_bayes.NaiveBayes`
              - Parameters removed from initialization but can be specified in fit(): categorical_varaible

            - **'RandomDecisionTree'** : :class:`~hana_ml.algorithms.pal.trees.RDTClassifier`

              - Disabled parameters: calculate_oob
              - Parameters removed from initialization but can be specified in fit(): categorical_variable, strata, priors

            - **'SVM'** : :class:`~hana_ml.algorithms.pal.svm.SVC`

              - Parameters removed from initialization but can be specified in fit(): categorical_variable

        An example for decision tree algorithm is shown below:

        You could create a dictionary to pass the arguments:

        >>> dt_params = dict(algorithm='c45',
                             model_format='json',
                             split_threshold=1e-5,
                             min_records_parent=2,
                             min_records_leaf=1,
                             thread_ratio=0.4,
                             resampling_method='cv',
                             evaluation_metric='auc',
                             fold_num=5,
                             progress_indicator_id='CV',
                             param_search_strategy = 'grid',
                             param_values = dict(split_threshold=[1e-3 , 1e-4, 1e-5])))

        >>> uni_dt = UnifiedClassification(func='DecisionTree', **dt_params),

        or use the following line instead as a whole:

        >>> uni_dt = UnifiedClassification(func='DecisionTree'
                                           algorithm='c45',
                                           model_format='json',
                                           split_threshold=1e-5,
                                           min_records_parent=2,
                                           min_records_leaf=1,
                                           thread_ratio=0.4,
                                           resampling_method='cv',
                                           evaluation_metric='auc',
                                           fold_num=5,
                                           progress_indicator_id='CV',
                                           param_search_strategy = 'grid',
                                           param_values = dict(split_threshold=[1e-3 , 1e-4, 1e-5]))

    Attributes
    ----------

    model_ : list of DataFrame.

        Model content.

    importance_ : DataFrame

        The feature importance (the higher, the more important the feature).

    statistics_ : DataFrame

        Names and values of statistics.

    optim_param_ : DataFrame

        Provides optimal parameters selected.

        Available only when parameter selection is triggered.

    confusion_matrix_ : DataFrame

        Confusion matrix used to evaluate the performance of classification
        algorithms.

    metrics_ : DataFrame

        Value of metrics.

    Examples
    --------
    Case1: Assume the training DataFrame is df_fit_rdt, data for prediction is df_predict and for score is df_score.

    Training the model:

    >>> rdt_params = dict(random_state=2,
                          split_threshold=1e-7,
                          min_samples_leaf=1,
                          n_estimators=10,
                          max_depth=55)

    >>> uc_rdt = UnifiedClassification(func = 'RandomDecisionTree', **rdt_params)

    >>> uc_rdt.fit(data=df_fit_rdt,
                   partition_method='stratified',
                   stratified_column='CLASS',
                   partition_random_state=2,
                   training_percent=0.7, ntiles=2)

    Output:

    >>> uc_rdt.importance_.collect().set_index('VARIABLE_NAME')
      VARIABLE_NAME  IMPORTANCE
    0       OUTLOOK    0.203566
    1          TEMP    0.479270
    2      HUMIDITY    0.317164
    3         WINDY    0.000000

    Prediction:

    >>> res = uc_rdt.predict(df_predict, key = "ID")[['ID', 'SCORE', 'CONFIDENCE']].collect()
       ID  SCORE  CONFIDENCE
    0   0   Play         1.0
    1   1   Play         0.8
    2   2   Play         0.7
    3   3   Play         0.9
    4   4   Play         0.8
    5   5   Play         0.8
    6   6   Play         0.9


    Score:

    >>> score_res = uc_rdt.score(data=self.df_score,
                                 key='ID',
                                 max_result_num=2,
                                 ntiles=2)[1]
    >>> score_res.head(4).collect()
       STAT_NAME         STAT_VALUE   CLASS_NAME
    0        AUC  0.673469387755102         None
    1     RECALL                  0  Do not Play
    2  PRECISION                  0  Do not Play
    3   F1_SCORE                  0  Do not Play


    Case 2: UnifiedReport for UnifiedClassification is shown as follows:

    >>> from hana_ml.algorithms.pal.model_selection import GridSearchCV
    >>> from hana_ml.algorithms.pal.model_selection import RandomSearchCV
    >>> hgc = UnifiedClassification('HybridGradientBoostingTree')
    >>> gscv = GridSearchCV(estimator=hgc,
                            param_grid={'learning_rate': [0.1, 0.4, 0.7, 1],
                                        'n_estimators': [4, 6, 8, 10],
                                        'split_threshold': [0.1, 0.4, 0.7, 1]},
                            train_control=dict(fold_num=5,
                                               resampling_method='cv',
                                               random_state=1,
                                               ref_metric=['auc']),
                            scoring='error_rate')
    >>> gscv.fit(data=diabetes_train, key= 'ID',
                 label='CLASS',
                 partition_method='stratified',
                 partition_random_state=1,
                 stratified_column='CLASS',
                 build_report=True)

    To look at the dataset report:

    >>> UnifiedReport(diabetes_train).build().display()

     .. image:: unified_report_dataset_report.png

    To see the model report:

    >>> UnifiedReport(gscv.estimator).display()

     .. image:: unified_report_model_report_classification.png

    We could also see the Optimal Parameter page:

     .. image:: unified_report_model_report_classification2.png

    """
    func_dict = {'decisiontree' : 'DT', 'logisticregression' : 'LOGR',
                 'multiclass-logisticregression' : 'M_LOGR',
                 'hybridgradientboostingtree' : 'HGBT',
                 'mlp' : 'MLP', 'naivebayes' : 'NB',
                 'randomforest' : 'RDT',
                 'randomdecisiontree' : 'RDT',
                 'svm' : 'SVM'}
    __cv_dict = {'resampling_method' : ('RESAMPLING_METHOD', str),
                 'evaluation_metric' : ('EVALUATION_METRIC', str),
                 'metric' : ('EVALUATION_METRIC', str),
                 'random_state' : ('SEED', int),
                 'fold_num' : ('FOLD_NUM', int),
                 'repeat_times' : ('REPEAT_TIMES', int),
                 'search_strategy' : ('PARAM_SEARCH_STRATEGY', str,
                                      {'random' : 'random', 'grid' : 'grid'}),
                 'param_search_strategy' : ('PARAM_SEARCH_STRATEGY', str,
                                            {'random' : 'random', 'grid' : 'grid'}),
                 'random_search_times' : ('RANDOM_SEARCH_TIMES', int),
                 'timeout' : ('TIMEOUT', int),
                 'progress_indicator_id' : ('PROGRESS_INDICATOR_ID', str),
                 'param_values' : ('_VALUES', dict),
                 'param_range' : ('_RANGE', dict)}
    __activation_map = {'tanh' : 1,
                        'linear' : 2,
                        'sigmoid_asymmetric' : 3,
                        'sigmoid_symmetric' : 4,
                        'gaussian_asymmetric' : 5,
                        'gaussian_symmetric' : 6,
                        'elliot_asymmetric' : 7,
                        'elliot_symmetric' : 8,
                        'sin_asymmetric' : 9,
                        'sin_symmetric' : 10,
                        'cos_asymmetric' : 11,
                        'cos_symmetric' : 12,
                        'relu' : 13}
    __fit_dict = {
        'DT' : {'bins' : ('_BIN_', ListOfTuples),
                'prior' : ('_PRIOR_', ListOfTuples)},
        'RDT' : {'strata' : ('STRATA', ListOfTuples),
                 'prior' : ('PRIOR', ListOfTuples)},
        'HGBT' : {},
        'LOGR' : {'class_map0' : ('CLASS_MAP0', str),
                  'class_map1' : ('CLASS_MAP1', str)},
        'MLP' : {},
        'NB' : {},
        'SVM' : {}}
    map_dict = {
        'DT' : {
            'algorithm' : ('ALGORITHM', int, dict(c45=1, chaid=2, cart=3)),
            'allow_missing_dependent' : ('ALLOW_MISSING_LABEL', bool),
            'percentage' : ('PERCENTAGE', float),
            'min_records_of_parent' : ('MIN_RECORDS_PARENT', int),
            'min_records_of_leaf' : ('MIN_RECORDS_LEAF', int),
            'max_depth' : ('MAX_DEPTH', int),
            'split_threshold' : ('SPLIT_THRESHOLD', float),
            'discretization_type' : ('DISCRETIZATION_TYPE', int, {'mdlpc' :0, 'equal_freq' :1}),
            #'bins' : ('_BIN_', ListOfTuples),
            'max_branch' : ('MAX_BRANCH', int),
            'merge_threshold' : ('MERGE_THRESHOLD', float),
            'use_surrogate' : ('USE_SURROGATE', bool),
            'model_format' : ('MODEL_FORMAT', int, {'json' :1, 'pmml' :2}),
            #'prior' : ('_PRIOR_', ListOfTuples),
            'thread_ratio' : ('THREAD_RATIO', float)},
        'RDT' : {
            'n_estimators' : ('N_ESTIMATORS', int),
            'max_features' : ('MAX_FEATURES', int),
            'min_samples_leaf' : ('MIN_SAMPLES_LEAF', int),
            'max_depth' : ('MAX_DEPTH', int),
            'split_threshold' : ('SPLIT_THRESHOLD', float),
            'random_state' : ('SEED', int),
            'thread_ratio' : ('THREAD_RATIO', float),
            'allow_missing_dependent' : ('ALLOW_MISSING_LABEL', bool),
            'sample_fraction' : ('SAMPLE_FRACTION', float),
            'compression' : ('COMPRESSION', bool),
            'max_bits' : ('MAX_BITS', int),
            'quantize_rate' : ('QUANTIZE_RATE', float),
            'model_format' : ('MODEL_FORMAT', int, {'json' :1, 'pmml' :2})},
        'HGBT' : {
            'n_estimators' : ('N_ESTIMATORS', int),
            'random_state' : ('SEED', int),
            'subsample' : ('SUBSAMPLE', float),
            'max_depth' : ('MAX_DEPTH', int),
            'split_threshold' : ('SPLIT_THRESHOLD', float),
            'learning_rate' : ('LEARNING_RATE', float),
            'split_method' : ('SPLIT_METHOD', str, {'exact' :'exact',
                                                    'sketch' :'sketch',
                                                    'sampling' :'sampling'}),
            'sketch_eps' : ('SKETCH_ESP', float),
            'min_sample_weight_leaf' : ('MIN_SAMPLES_WEIGHT_LEAF', float),
            'ref_metric' : ('REF_METRIC', ListOfStrings),
            'min_samples_leaf' : ('MIN_SAMPLES_LEAF', int),
            'max_w_in_split' : ('MAX_W_IN_SPLIT', float),
            'col_subsample_split' : ('COL_SUBSAMPLE_SPLIT', float),
            'col_subsample_tree' : ('COL_SUBSAMPLE_TREE', float),
            'lamb' : ('LAMBDA', float),
            'alpha' : ('ALPHA', float),
            'base_score' : ('BASE_SCORE', float),
            'adopt_prior' : ('START_FROM_AVERAGE', bool)},
        'LOGR' : {
            'multi_class' : ('M_', bool),
            'max_iter' : ('MAX_ITER', int),
            'enet_alpha' : ('ALPHA', float),
            'enet_lambda' : ('LAMB', float),
            'tol' : ('TOL', float),
            'solver' : ('METHOD', int, dict(auto=-1, newton=0, cyclical=2,
                                            lbfgs=3, stochastic=4, proximal=6)),
            'epsilon' : ('EPSILON', float),
            'standardize' : ('STANDARDIZE', bool),
            'max_pass_number' : ('MAX_PASS_NUMBER', int),
            'sgd_batch_number' : ('SGD_BATCH_NUMBER', int),
            'precompute' : ('PRECOMPUTE', bool),
            'handle_missing' : ('HANDLE_MISSING', bool),
            'lbfgs_m' : ('LBFGS_M', int),
            'stat_inf': ('STAT_INF', bool)},
        'MLP' : {
            'activation' : ('ACTIVATION',
                            int,
                            __activation_map),
            'activation_options' : ('ACTIVATION_OPTIONS', ListOfStrings),
            'output_activation' : ('OUTPUT_ACTIVATION',
                                   int,
                                   __activation_map),
            'output_activation_options' : ('OUTPUT_ACTIVATION_OPTIONS', ListOfStrings),
            'hidden_layer_size' : ('HIDDEN_LAYER_SIZE', (tuple, list)),
            'hidden_layer_size_options' : ('HIDDEN_LAYER_SIZE_OPTIONS', ListOfTuples),
            'max_iter' : ('MAX_ITER', int),
            'training_style' : ('TRAINING_STYLE', int, {'batch' : 0, 'stochastic' : 1}),
            'learning_rate' : ('LEARNING_RATE', float),
            'momentum' : ('MOMENTUM', float),
            'batch_size' : ('BATCH_SIZE', int),
            'normalization' : ('NORMALIZATION', int, {'no' : 0, 'z-transform' : 1, 'scalar' : 2}),
            'weight_init' : ('WEIGHT_INIT',
                             int,
                             {'all-zeros' : 0,
                              'normal' : 1,
                              'uniform' : 2,
                              'variance-scale-normal' : 3,
                              'variance-scale-uniform' : 4}),
            'thread_ratio' : ('THREAD_RATIO', float)},
        'NB' : {
            'alpha' : ('ALPHA', float),
            'discretization' : ('DISCRETIZATION', int, {'no' :0, 'supervised' :1}),
            'model_format' : ('MODEL_FORMAT', int, {'json' :0, 'pmml' :1}),
            'thread_ratio' : ('THREAD_RATIO', float)},
        'SVM' : {
            'c' : ('SVM_C', float),
            'kernel' : ('KERNEL_TYPE', int, {'linear' :0, 'poly' :1, 'rbf' :2, 'sigmoid' :3}),
            'degree' : ('POLY_DEGREE', int),
            'gamma' : ('RBF_GAMMA', float),
            'coef_lin' : ('COEF_LIN', float),
            'coef_const' : ('COEF_CONST', float),
            'probability' : ('PROBABILITY', bool),
            'shrink' : ('SHRINK', bool),
            'tol' : ('TOL', float),
            'evaluation_seed' : ('EVALUATION_SEED', int),
            'scale_info' : ('SCALE_INFO', int, {'no' :0, 'standardization' :1, 'rescale' :2}),
            'handle_missing' : ('HANDLE_MISSING', bool),
            'category_weight' : ('CATEGORY_WEIGHT', float),
            'compression' : ('COMPRESSION', bool),
            'max_bits' : ('MAX_BITS', int),
            'max_quantization_iter' : ('MAX_QUANTIZATION_ITER', int)}
    }
    #__partition_dict = dict(method=('PARTITION_METHOD', int, dict(no=0, user_defined=1, stratified=2)),
    #                        random_state=('PARTITION_RANDOM_SEED', int),
    #                        stratified_variable=('PARTITION_STRATIFIED_VARIABLE', str),
    #                        training_percent=('PARTITION_TRAINING_PERCENT', float),
    #                        training_size=('PARTITION_TRANING_SIZE', int))
    __partition_dict = dict(no=0, user_defined=1, stratified=2)

    def __init__(self,
                 func,
                 **kwargs):
        setattr(self, 'hanaml_parameters', pal_param_register())
        PALBase.__init__(self)
        _UnifiedClassificationReportBuilder.__init__(self, ["KEY", "VALUE"], ["KEY", "VALUE"])
        AMDPHelper.__init__(self)

        self.func = self._arg('Function name', func, self.func_dict)
        #params = self._arg('Function parameters', params, dict)
        self.params = dict(**kwargs)
        self.__real_func = self.func
        self.__pal_params = {}
        func_map = dict(self.map_dict[self.func], **self.__cv_dict)
        for parm in self.params:
            if parm in func_map.keys():
                parm_val = self.params[parm]
                if self.func == 'LOGR' and parm == 'multi_class' and parm_val is True:#pylint:disable=line-too-long
                    self.__real_func = 'M_LOGR'
                arg_map = func_map[parm]
                if arg_map[1] == ListOfStrings and isinstance(parm_val, str):
                    parm_val = [parm_val]
                if len(arg_map) == 2:
                    self.__pal_params[arg_map[0]] = (self._arg(parm, parm_val, arg_map[1]), arg_map[1])
                else:
                    self.__pal_params[arg_map[0]] = (self._arg(parm, parm_val, arg_map[2]), arg_map[1])
            else:
                err_msg = "'{}' is not a valid parameter name for initializing a {} model".format(parm, func)
                logger.error(err_msg)
                raise KeyError(err_msg)
        if self.__real_func == 'M_LOGR' and 'METHOD' in self.__pal_params:
            if self.__pal_params['METHOD'][0] in (None, -1):
                self.__pal_params['METHOD'] = (None, int)
            elif self.__pal_params['METHOD'][0] == 3:
                self.__pal_params['METHOD'] = (0, int)
            elif self.__pal_params['METHOD'][0] == 2:
                self.__pal_params['METHOD'] = (1, int)
            else:
                msg = "Solver not supported for multi-class logistic regression."
                logger.error(msg)
                raise ValueError(msg)
        self.__param_rows = None

    def update_cv_params(self, name, value, typ):
        """
        Update parameters for model-evaluation/parameter-selection.
        """
        if name in self.__cv_dict.keys():
            self.__pal_params[self.__cv_dict[name][0]] = (value, typ)

    def __map_param(self, name, value, typ,
                    label_type='NVARCHAR'):
        tpl = ()
        if typ in [int, bool]:
            tpl = (name, value, None, None)
        elif typ == float:
            tpl = (name, None, value, None)
        elif typ in [str, ListOfStrings]:
            tpl = (name, None, None,
                   value.upper() if '_METRIC' in name else value)
        else:
            if self.func == 'RDT':
                if label_type in ['VARCHAR', 'NVARCHAR']:
                    tpl = (name, None, value[1], value[0])
                else:
                    tpl = (name, value[0], value[1], None)
            elif self.func == 'DT':
                if name == '_BIN_':
                    tpl = (str(value[0])+name, value[1], None, None)
                else:
                    tpl = (str(value[0])+name, None, value[1], None)
        return tpl

    @trace_sql
    def fit(self,#pylint: disable=too-many-branches, too-many-statements
            data,
            key=None,
            features=None,
            label=None,
            purpose=None,
            partition_method=None,
            stratified_column=None,
            partition_random_state=None,
            training_percent=None,
            training_size=None,
            ntiles=None,
            categorical_variable=None,
            output_partition_result=None,
            background_size=None,
            background_random_state=None,
            build_report=False,
            **kwargs):
        """
        Fit function for unified classification.

        Parameters
        ----------

        data : DataFrame
            Training data.

        key : str, optional
            Name of the ID column.

            If ``key`` is not provided, then:

                - if ``data`` is indexed by a single column, then ``key`` defaults
                  to that index column;
                - otherwise, it is assumed that ``data`` contains no ID column.

        features : list of str, optional
            Names of the feature columns.

            If ``features`` is not provided, it defaults to all non-ID,
            non-label columns.

        label : str, optional
            Name of the dependent variable.
            If ``label`` is not provided, it defaults to the last non-ID column.

        purpose : str, optional
            Indicates the name of purpose column which is used for user self-defined data partition.

            The meaning of value in the column for each data instance is shown below:
                - 1 : training
                - 2 : validation

            Valid and mandatory only when ``partition_method`` is 'user_defined'.

            No default value.

        partition_method : {'no', 'user_defined', 'stratified'}, optional
            Defines the way to divide the dataset.
                - 'no' : no partition.
                - 'user_defined' : user defined  partition.
                - 'stratified' : stratified partition.

            Defaults to 'no'.
        stratified_column : str, optional
            Indicates which column is used for stratification.

            Valid only when ``partition_method`` is set to 'stratified'.

            No default value.
        partition_random_state : int, optional
            Indicates the seed used to initialize the random number generator.

            Valid only when ``partition_method`` is set to 'stratified'.
                - 0 : Uses the system time.
                - Not 0 : Uses the specified seed.

            Defaults to 0.
        training_percent : float, optional
            The percentage of data used for training.
            Value range: 0 <= value <= 1.

            Defaults to 0.8.
        training_size : int, optional
            Row size of data used for training. Value range >= 0.

            If both ``training_percent`` and ``training_size`` are specified, ``training_percent`` takes precedence.

            No default value.
        ntiles : int, optional
            Used to control the population tiles in metrics output.

            The value should be at least 1 and no larger than the row size of the input data

            If the row size of data for metrics evaluation is less than 20,
            the default value is 1; otherwise it is 20.

        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            No default value.

        output_partition_result : bool, optional
            Specifies whether or not to output the partition result.

            Valid only when ``partition_method`` is not 'no', and ``key`` is not None.

            Defaults to False.

        background_size : int, optional
            Specifies the size of background data used for Shapley Additive Explanations (SHAP) values calculation.

            Should not larger than the size of training data.

            Valid only for Naive Bayes, Suppoert Vector Machine, or Multilayer Perceptron and Multi-class
            Logistic Regression models.

            Defaults to 0(no background data, in which case the calculation of SHAP values shall be disabled).

        background_random_state : int, optional
            Specifies the seed for random number generator in the background data samping.

                - 0 : Uses current time as seed
                - Others : The specified seed value

            Valid only for Naive Bayes, Suppoert Vector Machine, or Multilayer Perceptron and Multi-class
            Logistic Regression models.

            Defaults to 0.

        build_report : bool, optional
            Whether to build report or not.

            Defaults to False.

        **kwargs : keyword arguments
            Additional keyword arguments of model fitting for different classification algorithms.

            Please referred to the fit function of each algorithm as follows:
                - **'DecisionTree'** : :class:`~hana_ml.algorithms.pal.trees.DecisionTreeClassifier`
                - **'HybridGradientBoostingTree'** : :class:`~hana_ml.algorithms.pal.trees.HybridGradientBoostingClassifier`
                - **'LogisticRegression'** :class:`~hana_ml.algorithms.pal.linear_model.LogisticRegression`
                - **'MLP'** : :class:`~hana_ml.algorithms.pal.neural_network.MLPClassifier`
                - **'NaiveBayes'** : :class:`~hana_ml.algorithms.pal.naive_bayes.NaiveBayes`
                - **'RandomDecisionTree'** : :class:`~hana_ml.algorithms.pal.trees.RDTClassifier`
                - **'SVM'** : :class:`~hana_ml.algorithms.pal.svm.SVC`

        Returns
        -------

        Fitted object.
        """
        setattr(self, 'hanaml_fit_params', pal_param_register())
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
        partition_method = self._arg('partition_method', partition_method, self.__partition_dict)
        stratified_column = self._arg('stratified_column', stratified_column, str, required=partition_method == 2)
        partition_random_state = self._arg('partition_random_state', partition_random_state, int)
        training_percent = self._arg('training_percent', training_percent, float)
        training_size = self._arg('training_percent', training_size, int)
        output_partition_result = self._arg('output_partition_result',
                                            output_partition_result, bool)
        background_size = self._arg('background_size', background_size, int)
        background_random_state = self._arg('background_random_state', background_random_state, int)
        check = False
        if partition_method is not None:
            if partition_method == 1:#pylint:disable=line-too-long
                check = True
        if check is False:
            purpose = None
        purpose = self._arg('purpose', purpose, str, check)
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        cols = data.columns
        if key is not None:
            id_col = [key]
            cols.remove(key)
        else:
            id_col = []
        if purpose is not None:
            cols.remove(purpose)
            purpose = [purpose]
        else:
            purpose = []
        if label is None:
            label = cols[-1]
        cols.remove(label)
        label_t = data.dtypes([label])[0][1]
        if features is None:
            features = cols
        data_ = data[id_col + features + [label] + purpose]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        param_rows = [('FUNCTION', None, None, self.__real_func),
                      ('KEY', key is not None, None, None),
                      ('PARTITION_METHOD', partition_method, None, None),
                      ('PARTITION_RANDOM_SEED', partition_random_state, None, None),
                      ('PARTITION_STRATIFIED_VARIABLE',
                       None, None, stratified_column),
                      ('PARTITION_TRAINING_PERCENT', None, training_percent, None),
                      ('PARTITION_TRAINING_SIZE', training_size, None, None),
                      ('OUTPUT_PARTITION_RESULT', output_partition_result,
                       None, None),
                      ('BACKGROUND_SIZE', background_size, None, None),
                      ('BACKGROUND_SAMPLING_SEED', background_random_state,
                       None, None)]
        for name in self.__pal_params:
            value, typ = self.__pal_params[name]
            if isinstance(value, (list, tuple)):
                if name == 'HIDDEN_LAYER_SIZE':
                    value = ', '.join([str(v) for v in value])
                    param_rows.extend([(name, None, None, value)])
                elif name == 'HIDDEN_LAYER_SIZE_OPTIONS':
                    value = ', '.join([str(v) for v in value])
                    value = value.replace('(', '"').replace(')', '"')
                    value = value.replace('[', '"').replace(']', '"')
                    value = '{' + value + '}'
                    param_rows.extend([(name, None, None, value)])
                elif name in ['ACTIVATION_OPTIONS', 'OUTPUT_ACTIVATION_OPTIONS']:
                    value = ', '.join([str(self.__activation_map[v]) for v in value])
                    value = '{' + value +'}'
                    param_rows.extend([(name, None, None, value)])
                else:
                    for val in value:
                        tpl = [self.__map_param(name, val, typ, label_t)]
                        param_rows.extend(tpl)
            elif typ == dict:
                if name == '_RANGE':
                    for var in value:
                        rge = [str(v) for v in value[var]]
                        rge_str = '[' + ((',' if len(rge) == 3 else ',,'). join(rge)) + ']'
                        tpl = [(self.map_dict[self.func][var][0] + name, None, None, rge_str)]
                        param_rows.extend(tpl)
                elif name == '_VALUES':
                    for var in value:
                        vvr = [str(v) for v in value[var]]
                        vvr_str = '{' + ','.join(vvr) + '}'
                        tpl = [(self.map_dict[self.func][var][0] + name, None, None, vvr_str)]
                        param_rows.extend(tpl)
            else:
                tpl = [self.__map_param(name, value, typ, label_t)]
                param_rows.extend(tpl)
        ntiles = self._arg('ntiles', ntiles, int)
        if ntiles is not None:
            param_rows.extend([('NTILES', ntiles, None, None)])
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        if 'INT' in label_t:
            if categorical_variable is None:
                categorical_variable = [label]
            elif label not in categorical_variable and self.func != 'LOGR':
                categorical_variable = categorical_variable.append(label)
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, var) for var in categorical_variable])
        add_param_map = self.__fit_dict[self.func]
        for kye in kwargs:
            if kye in add_param_map.keys():
                par_val = kwargs[kye]
                if add_param_map[kye][1] == ListOfStrings and isinstance(par_val, str):
                    par_val = [par_val]
                value = self._arg(kye, par_val, add_param_map[kye][1])
                if isinstance(value, (list, tuple)):
                    for val in value:
                        tpl = [self.__map_param(add_param_map[kye][0],
                                                val,
                                                add_param_map[kye][1])]
                        param_rows.extend(tpl)
                if kye in ['class_map0', 'class_map1']:
                    param_rows.extend([(kye.upper(), None, None, kwargs[kye])])
            else:
                err_msg = "'{}' is not a valid parameter for fitting the classification model.".format(kye)
                logger.error(err_msg)
                raise KeyError(err_msg)
        if self.__real_func == 'LOGR'  and any([var not in kwargs for var in ['class_map0', 'class_map1']]):
            if label_t in ['VARCHAR', 'NVARCHAR']:
                err_msg = 'Values of class_map0 and class_map1 must be specified when fitting LOGR model.'
                logger.error(err_msg)
                raise ValueError(err_msg)

            #param_rows.extend([('CLASS_MAP0', None, None, '0'), ('CLASS_MAP1', None, None, '1')])
        outputs = ['MODEL', 'IMPORTANCE', 'STATS', 'OPT_PARAM', 'CONFUSION_MATRIX',
                   'METRICS', 'PARTITION_TYPE', 'PLACE_HOLDER2']
        outputs = ['#PAL_UNIFIED_CLASSIFICATION_{}_{}_{}'.format(tbl, self.id, unique_id)
                   for tbl in outputs]
        model_tbl, imp_tbl, stats_tbl, opt_param_tbl, cm_tbl, metrics_tbl, partition_tbl, _ = outputs
        self.fit_data = data_
        self.label = label
        if self.label is None:
            self.label = data_.columns[-1]
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_CLASSIFICATION',
                                data_,
                                ParameterTable().with_data(param_rows),
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
        self.importance_ = conn.table(imp_tbl)
        self.statistics_ = conn.table(stats_tbl)
        self.optimal_param_ = conn.table(opt_param_tbl)
        self.confusion_matrix_ = conn.table(cm_tbl)
        self.metrics_ = conn.table(metrics_tbl)
        self.partition_type = conn.table(partition_tbl)
        self.model_ = [conn.table(model_tbl),
                       self.statistics_,
                       self.optimal_param_,
                       self.confusion_matrix_,
                       self.importance_,
                       self.metrics_]
        self.__param_rows = param_rows
        if build_report:
            self.build_report()
        return self

    @trace_sql
    def predict(self, data,#pylint: disable=too-many-branches, invalid-name, too-many-statements
                key=None,
                features=None,
                model=None,
                thread_ratio=None,
                verbose=None,
                class_map1=None,#pylint:disable=invalid-name
                class_map0=None,#pylint:disable=invalid-name
                alpha=None,
                block_size=None,
                missing_replacement=None,
                categorical_variable=None,
                top_k_attributions=None,
                attribution_method=None,
                sample_size=None,
                random_state=None,
                **kwargs):
        r"""
        Predict with the classification model.

        Parameters
        ----------
        data :  DataFrame
            Data to be predicted.

        key : str, optional
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : ListOfStrings, optional
            Names of feature columns in data for prediction.

            Defaults all non-ID columns in `data` if not provided.

        model : DataFrame
            Fitted classification model.

            Defaults to self.model\_.

        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to the PAL's default value.

        verbose : bool, optional
            Specifies whether to output all classes and the corresponding confidences for each data.

            Defaults to False.

        class_map0 : str, optional
            Specifies the label value which will be mapped to 0 in logistic regression.

            Valid only for logistic regression models when label variable is of VARCHAR or NVARCHAR type.

            No default value.

        class_map1 : str, optional
            Specifies the label value which will be mapped to 1 in logistic regression.

            Valid only for logistic regression models when label variable is of VARCHAR or NVARCHAR type.

            No default value.

        alpha : float, optional
            Specifies the value for laplace smoothing.

                - 0: Disables Laplace smoothing.
                - Other positive values: Enables Laplace smoothing for discrete values.

            Valid only for Naive Bayes models.

            Defaults to 0.

        block_size : int, optional
            Specifies the number of data loaded per time during scoring.

                - 0: load all data once
                - Other positive Values: the specified number

            Valid only for RandomDecisionTree and HybridGradientBoostingTree model

            Defaults to 0.

        missing_replacement : str, optional
            Specifies the strategy for replacement of missing values in prediction data.

                - 'feature_marginalized': marginalises each missing feature out independently
                - 'instance_marginalized': marginalises all missing features in an instance as a
                  whole corresponding to each category

            Valid only for RandomDecisionTree and HybridGradientBoostingTree models.

            Defaults to 'feature_marginalized'.

        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            No default value.

        top_k_attributions : int, optional
            Specifies the number of features with highest attributions to output.

            Defaults to 10.

        attribution_method : {'no', 'saabas', 'tree-shap'}, optional

            Specifies which method to use for model reasoning.

                - 'no' : No reasoning
                - 'saabas' : Saabas method
                - 'tree-shap' : Tree SHAP method

            Valid only for tree-based models, i.e. DecisionTree, RandomDecisionTree and HybridGradientBoostingTree models.

            Defaults to 'tree-shap'.

        sample_size : int, optional
            Specifies the number of sampled combinations of features.

                - 0 : Heuristically determined by algorithm
                - Others : The specified sample size

            Valid only for Naive Bayes, Support Vector Machine and Multilayer Perceptron models.

            Defaults to 0.

        random_state : int, optional
            Specifies the seed for random number generator when sampling the combination of features.

                - 0 : User current time as seed
                - Others : The actual seed

            Valid only for Naive Bayes, Support Vector Machine and Multilayer Perceptron models.

            Defaults to 0.

        **kwargs : keyword arguments
            Additional keyword arguments w.r.t. different classification algorithms within UnifiedClassification.

        Returns
        -------
            DataFrame
                - Predicted result, structured as follows:

                    -  1st column : ID
                    -  2nd column : SCORE, i.e. predicted class label
                    -  3rd column : CONFIDENCE, i.e. confidence value for the assigned class label
                    -  4th column : REASON CODE, valid only for tree-based functionalities.
        """
        if not hasattr(self, 'model_') or getattr(self, 'model_') is None:
            if model is None:
                raise FitIncompleteError("Model not initialized. Perform a fit first.")
        conn = data.connection_context
        method_map = {'no': 0, 'saabas': 1, 'tree-shap': 2}
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
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        if features is None:
            features = cols
        data_ = data[[key] + features]
        thread_ratio = self._arg('thread_ratio', thread_ratio, float)
        verbose = self._arg('verbose', verbose, bool)
        class_map0 = self._arg('class_map0', class_map0, str)
        class_map1 = self._arg('class_map1', class_map1, str)
        alpha = self._arg('alpha', alpha, float)
        block_size = self._arg('block_size', block_size, int)
        replacement_map = dict(feature_marginalized=1, instance_marginalized=2)
        missing_replacement = self._arg('missing_replacement', missing_replacement,
                                        replacement_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        top_k_attributions = self._arg('top_k_attributions', top_k_attributions, int)
        attribution_method = self._arg('attribution_method', attribution_method, method_map)
        sample_size = self._arg('sample_size', sample_size, int)
        random_state = self._arg('random_state', random_state, int)
        if model is None:
            if isinstance(self.model_, (list, tuple)):
                model = self.model_[0]
            else:
                model = self.model_
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['#PAL_UNIFIED_CLASSIF_PREDICT_{}_TBL_{}_{}'.format(tb, self.id, unique_id) for tb in ['RESULT', 'PH']]
        param_rows = [('FUNCTION', None, None, self.__real_func),
                      ('CLASS_MAP0', None, None, class_map0),
                      ('CLASS_MAP1', None, None, class_map1),
                      ('LAPLACE', None, alpha, None),
                      ('BLOCK_SIZE', block_size, None, None),
                      ('MISSING_REPLACEMENT', missing_replacement, None, None),
                      ('TOP_K_ATTRIBUTIONS', top_k_attributions, None, None),
                      ('FEATURE_ATTRIBUTION_METHOD', attribution_method, None, None),
                      ('SAMPLESIZE', sample_size, None, None),
                      ('SEED', random_state, None, None)]
        if thread_ratio is not None:
            param_rows.extend([('THREAD_RATIO', None, thread_ratio, None)])
        if verbose is not None:
            param_rows.extend([('VERBOSE', verbose, None, None)])
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, var) for var in categorical_variable])
        param = dict(**kwargs)
        if not param:
            func_map = self.map_dict[self.func]
            for name, value in param.items():
                name_type = func_map[name]
                name_ = name_type[0]
                typ = name_type[1]
                if len(name_type) == 3:
                    value = name_type[2][value]
                if isinstance(value, (list, tuple)):
                    for val in value:
                        param_rows.extend([self.__map_param(name_, val, typ)])
                else:
                    param_rows.extend([self.__map_param(name_, value, typ)])
        self.predict_data = data_
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_CLASSIFICATION_PREDICT',
                                data_,
                                model,
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
        return conn.table(outputs[0])

    def score(self, data,#pylint:disable=invalid-name, too-many-statements, too-many-branches
              key=None,
              features=None,
              label=None,
              model=None,
              thread_ratio=None,
              max_result_num=None,
              ntiles=None,
              class_map1=None,#pylint:disable=invalid-name
              class_map0=None,#pylint:disable=invalid-name
              alpha=None,
              block_size=None,
              missing_replacement=None,
              categorical_variable=None,
              top_k_attributions=None,
              attribution_method=None,
              sample_size=None,
              random_state=None):
        r"""
        Users can use the score function to evaluate the model quality.
        In the Unified Classification, statistics and metrics are provided to show the model quality.
        Currently the following metrics are supported.

            - AUC and ROC
            - RECALL, PRECISION, F1-SCORE, SUPPORT
            - ACCURACY
            - KAPPA
            - MCC
            - CUMULATIVE GAINS
            - CULMULATIVE LIFT
            - Lift

        Parameters
        ----------
        data :  DataFrame
            Data for scoring.

        key : str, optional
            Name of the ID column.

            Mandatory if ``data`` is not indexed, or the index of ``data`` contains multiple columns.

            Defaults to the single index column of ``data`` if not provided.

        features : ListOfString or str, optional
            Names of feature columns.

            Defaults to all non-ID, non-label columns if not provided.

        label : str, optional
            Name of the label column.

            Defaults to the last non-ID column if not provided.

        model : DataFrame
            Fitted classification model.

            Defaults to self.model\_.

        thread_ratio : float, optional
            Controls the proportion of available threads to use for prediction.

            The value range is from 0 to 1, where 0 indicates a single thread,
            and 1 indicates up to all available threads.

            Values between 0 and 1 will use that percentage of available threads.

            Values outside this range tell PAL to heuristically determine the number of threads to use.

            Defaults to the PAL's default value.

        max_result_num : int, optional
            Specifies the output number of prediction results.

        label : str, optional
            The setting of the parameter should be same with the one in train.

        ntiles : int, optional
            Used to control the population tiles in metrics output.

            The value should be at least 1 and no larger than the row size of the input data

        class_map0 : str, optional
            Specifies the label value which will be mapped to 0 in logistic regression.

            Valid only for logistic regression models when label variable is of VARCHAR or NVARCHAR type.

            No default value.

        class_map1 : str, optional
            Specifies the label value which will be mapped to 1 in logistic regression.

            Valid only for logistic regression models when label variable is of VARCHAR or NVARCHAR type.

            No default value.

        alpha : float, optional
            Specifies the value for laplace smoothing.

                - 0: Disables Laplace smoothing.
                - Other positive values: Enables Laplace smoothing for discrete values.

            Valid only for Naive Bayes models.

            Defaults to 0.

        block_size : int, optional
            Specifies the number of data loaded per time during scoring.

                - 0: load all data once
                - Other positive Values: the specified number

            Valid only for RandomDecisionTree and HybridGradientBoostingTree models.

            Defaults to 0.

        missing_replacement : str, optional
            Specifies the strategy for replacement of missing values in prediction data.

                - 'feature_marginalized': marginalises each missing feature out independently
                - 'instance_marginalized': marginalises all missing features in an instance as a \
                  whole corresponding to each category

            Defaults to 'feature_marginalized'.

        categorical_variable : str or list of str, optional
            Specifies INTEGER column(s) that should be treated as categorical.

            Other INTEGER columns will be treated as continuous.

            Valid only for logistic regression models.

            No default value.

        top_k_attributions : int, optional
            Specifies the number of features with highest attributions to output.

            Defaults to 10.

        attribution_method : {'no', 'saabas', 'tree-shap'}, optional

            Specifies which method to use for model reasoning.

                - 'no' : No reasoning
                - 'saabas' : Saabas method
                - 'tree-shap' : Tree SHAP method

            Valid only for tree-based models, i.e. DecisionTree, RandomDecisionTree and HybridGradientBoostingTree models.

            Defaults to 'tree-shap'.

        sample_size : int, optional
            Specifies the number of sampled combinations of features.

                - 0 : Heuristically determined by algorithm
                - Others : The specified sample size

            Valid only for Naive Bayes, Support Vector Machine and Multilayer Perceptron models.

            Defaults to 0.

        random_state : int, optional
            Specifies the seed for random number generator when sampling the combination of features.

                - 0 : User current time as seed
                - Others : The actual seed

            Valid only for Naive Bayes, Support Vector Machine and Multilayer Perceptron models.

            Defaults to 0.

        Returns
        -------
            DataFrame
                - Prediction result by ignoring the true labels of the input data,
                  structured the same as the result table of predict() function.
                - Statistics.
                - Confusion matrix.
                - Metrics.
        """
        if not hasattr(self, 'model_') or getattr(self, 'model_') is None:
            if model is None:
                raise FitIncompleteError("Model not initialized. Perform a fit first.")
        conn = data.connection_context
        require_pal_usable(conn)
        method_map = {'no': 0, 'saabas': 1, 'tree-shap': 2}
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
        if isinstance(features, str):
            features = [features]
        features = self._arg('features', features, ListOfStrings)
        label = self._arg('label', label, str)
        if label is None:
            label = cols[-1]
        cols.remove(label)
        if features is None:
            features = cols
        thread_ratio = self._arg('thread_ratio', thread_ratio,
                                 float)
        max_result_num = self._arg('max_result_num',
                                   max_result_num, int)
        ntiles = self._arg('ntiles', ntiles, int)
        class_map0 = self._arg('class_map0', class_map0, str)
        class_map1 = self._arg('class_map1', class_map1, str)
        alpha = self._arg('alpha', alpha, float)
        block_size = self._arg('block_size', block_size, int)
        replacement_map = dict(feature_marginalized=1, instance_marginalized=2)
        missing_replacement = self._arg('missing_replacement', missing_replacement,
                                        replacement_map)
        if isinstance(categorical_variable, str):
            categorical_variable = [categorical_variable]
        categorical_variable = self._arg('categorical_variable', categorical_variable, ListOfStrings)
        top_k_attributions = self._arg('top_k_attributions', top_k_attributions, int)
        attribution_method = self._arg('attribution_method', attribution_method, method_map)
        sample_size = self._arg('sample_size', sample_size, int)
        random_state = self._arg('random_state', random_state, int)
        data_ = data[[key] + features + [label]]
        if model is None:
            if isinstance(self.model_, (list, tuple)):
                model = self.model_[0]
            else:
                model = self.model_
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        outputs = ['RESULT', 'STATS', 'CONFUSION_MATRIX', 'METRICS']
        outputs = ['#PAL_UNIFIED_CLASSIFICATION_SCORE_{}_TBL_{}_{}'.format(name, self.id, unique_id)
                   for name in outputs]
        result_tbl, stats_tbl, cm_tbl, metrics_tbl = outputs
        param_rows = [('FUNCTION', None, None, self.__real_func),
                      ('THREAD_RATIO', None, thread_ratio, None),
                      ('MAX_RESULT_NUM', max_result_num, None, None),
                      ('NTILES', ntiles, None, None),
                      ('CLASS_MAP0', None, None, class_map0),
                      ('CLASS_MAP1', None, None, class_map1),
                      ('LAPLACE', None, alpha, None),
                      ('BLOCK_SIZE', block_size, None, None),
                      ('MISSING_REPLACEMENT', missing_replacement, None, None),
                      ('TOP_K_ATTRIBUTIONS', top_k_attributions, None, None),
                      ('FEATURE_ATTRIBUTION_METHOD', attribution_method, None, None),
                      ('SAMPLESIZE', sample_size, None, None),
                      ('SEED', random_state, None, None)]
        if categorical_variable is not None:
            param_rows.extend([('CATEGORICAL_VARIABLE', None, None, var) for var in categorical_variable])
        # SQLTRACE
        conn.sql_tracer.set_sql_trace(self, self.__class__.__name__, 'Score')
        try:
            self._call_pal_auto(conn,
                                'PAL_UNIFIED_CLASSIFICATION_SCORE',
                                data_,
                                model,
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
        return conn.table(result_tbl), conn.table(stats_tbl), conn.table(cm_tbl), conn.table(metrics_tbl)

    def build_report(self):
        """
        Build model report.
        """
        try:
            rowlist = []
            for key, val in self.hanaml_parameters["kwargs"].items():
                rowlist.append({"KEY": key, "VALUE": str(val)})
            parameter_df = pd.DataFrame(rowlist)
            # pylint: disable=protected-access
            self._set_statistic_table(self.model_[1]) \
                ._set_parameter_table(parameter_df) \
                ._set_optimal_parameter_table(self.model_[2].collect()) \
                ._set_confusion_matrix_table(self.model_[3]) \
                ._set_variable_importance_table(self.model_[4]) \
                ._set_metric_table(self.model_[5]) \
                ._render_report()
        except Exception as err:
            logger.error(str(err))
            raise
        return self

    def create_amdp_class(self,
                          amdp_name,
                          training_dataset='',
                          apply_dataset='',
                          num_reason_features=3):
        """
        Create AMDP class file. Then build_amdp_class can be called to generate amdp class.

        Parameters
        ----------
        training_dataset : str, optional
            Name of training dataset.

            Defaults to ''.
        apply_dataset : str, optional
            Name of apply dataset.

            Defaults to ''.
        num_reason_features : int, optional
            The number of features that contribute to the classification decision the most.
            This reason code information is to be displayed during the prediction phase.

            Defaults to 3.
        """
        self.add_amdp_template("tmp_hemi_unified_classification_func.abap")
        self.add_amdp_name(amdp_name)
        self.load_abap_class_mapping()
        fit_data_struct = ''
        fit_data_st = {}
        if hasattr(self, "fit_data_struct"):
            fit_data_st = self.fit_data_struct
        if hasattr(self, "fit_data"):
            if self.fit_data:
                fit_data_st = self.fit_data.get_table_structure()
        if fit_data_st.keys():
            for key, val in fit_data_st.items():
                fit_data_struct = fit_data_struct + " " * 8 + "{} TYPE {},\n".format(key.lower(),
                                                                                     self.abap_class_mapping(val))
            self.add_amdp_item("<<TRAIN_INPUT_STRUCTURE>>",
                               fit_data_struct[:-1])
        self.add_amdp_item("<<CAST_TARGET_OUTPUT>>", '')
        self.add_amdp_item("<<RESULT_OUTPUT_STRUCTURE>>",
                           " " * 8 + "id TYPE int4,\n" +\
                           " " * 8 + "score TYPE int4,\n" +\
                           " " * 8 + "confidence TYPE f,")
        reasoncode_struct = ''
        for num in range(0, num_reason_features):
            reasoncode_struct = reasoncode_struct + " " * 8 +\
                        "reason_code_feature_{} TYPE shemi_reason_code_feature_name,\n".format(num + 1) +\
                        " " * 8 +"reason_code_percentage_{} TYPE shemi_reason_code_feature_pct,\n".format(num + 1)
        self.add_amdp_item("<<REASON_CODE_STRUCTURE>>",
                           reasoncode_struct[:-1])
        self.add_amdp_item("<<TRAINING_DATASET>>",
                           training_dataset)
        self.add_amdp_item("<<APPLY_DATASET>>",
                           apply_dataset)
        param_meta = []
        param_default_meata = []
        for fit_param in self.get_fit_parameters():
            param_meta.append("( name = '{}' type = cl_hemi_constants=>cs_param_type-string role = cl_hemi_constants=>cs_param_role-train configurable = abap_true has_context = abap_false )".format(fit_param[0]))
            param_default_meata.append("( name = '{}' value = '{}' )".format(fit_param[0], fit_param[1]))
        if self.get_predict_parameters():
            for predict_param in self.get_predict_parameters():
                param_meta.append("name = '{}' type = cl_hemi_constants=>cs_param_type-string role = cl_hemi_constants=>cs_param_role-apply configurable = abap_true has_context = abap_false )".format(predict_param[0]))
                param_default_meata.append("( name = '{}' value = '{}' )".format(predict_param[0], predict_param[1]))
        self.add_amdp_item("<<PARAMETER>>",
                           "( {} )".format("\n".join(param_meta)))
        self.add_amdp_item("<<PARAMETER_DEFAULT>>",
                           "( {} )".format("\n".join(param_default_meata)))
        self.add_amdp_item("<<TARGET_COLUMN>>",
                           self.label)
        self.add_amdp_item("<<KEY_FIELD_DESCRIPTION>>",
                           '')
        predict_data_cols = ''
        predict_data_st = {}
        if hasattr(self, "predict_data_struct"):
            predict_data_st = self.predict_data_struct
        if hasattr(self, "predict_data"):
            if self.predict_data:
                predict_data_st = self.predict_data.get_table_structure()
        if predict_data_st.keys():
            for key, val in predict_data_st.items():
                predict_data_cols = predict_data_cols + " " * 16 + "{},\n".format(key.lower())
            self.add_amdp_item("<<PREDICT_DATA_COLS>>",
                               predict_data_cols[:-2])
        result_field = []
        result_field.append('cast(result.ID as "$ABAP.type( INT4 )") as ID,')
        result_field.append(" " * 23 + 'cast(result.SCORE as "$ABAP.type( INT4 )") as SCORE,')
        result_field.append(" " * 23 + 'cast(result.CONFIDENCE as "$ABAP.type( FLTP )") as CONFIDENCE,')
        self.add_amdp_item("<<RESULT_FIELDS>>",
                           "\n".join(result_field)[:-1])
        reasoncode_result = ''
        for num in range(0, num_reason_features):
            reasoncode_result = reasoncode_result + " " * 23 +\
                        "trim(both '\"' from json_query(result.reason_code, '$[{}].attr')) as reason_code_feature_{},\n".format(num, num + 1) +\
                        " " * 23 +"json_query(result.reason_code, '$[{}].pct' ) as reason_code_percentage_{},\n".format(num, num + 1)
        self.add_amdp_item("<<RESULT_REASON_CODE_FIELDS>>",
                           reasoncode_result[:-2])
        return self
