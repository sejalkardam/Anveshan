#pylint: disable=invalid-name, too-many-arguments, too-many-locals, line-too-long
#pylint: disable=too-many-lines, relative-beyond-top-level, too-many-arguments, bare-except
#pylint: disable=superfluous-parens, too-many-branches, no-else-return, broad-except
#pylint: disable=consider-using-f-string
'''
This module provides various functions of text minig. The following functions are available:
    * :func:`tf_analysis`
    * :func:`text_classification`
    * :func:`get_related_doc`
    * :func:`get_related_term`
    * :func:`get_relevant_doc`
    * :func:`get_relevant_term`
    * :func:`get_suggested_term`
    * :class:`TFIDF`
'''
import os
import logging
import uuid
import time
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml.dataframe import quotename, DataFrame
from hana_ml.ml_base import try_drop, execute_logged
from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    arg,
    PALBase,
    call_pal_auto_with_hint,
    require_pal_usable
)

logger = logging.getLogger(__name__)

def tf_analysis(data, lang=None):
    """
    Perform Term Frequency(TF) analysis on the given document.
    TF is the number of occurrences of term in document.

    Parameters
    ----------
    data : DataFrame
        - 1st column, ID.
        - 2nd column, Document content.
        - 3rd column, Document category.

    lang : str, optional
        Specify the language type.

        Defaults to 'EN'.

    Returns
    -------
    A tuple of DataFrame
        TF-IDF result, structured as follows:
            - TM_TERM.
            - TM_TERM_FREQUENCY.
            - TM_IDF_FREQUENCY.
            - TF_VALUE.
            - IDF_VALUE.
            - TF_IDF_VALUE.

        Document term frequency table, structured as follows:
            - ID.
            - TM_TERM.
            - TM_TERM_FREQUENCY.

        Document category table, structured as follows:
            - ID.
            - Document category.

    Examples
    --------

    The input DataFrame df:

    >>> df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    Invoke tf_analysis function:

    >>> tfidf= tf_analysis(df)

    Output:

    >>> tfidf[0].head(3).collect()
      TM_TERMS TM_TERM_TF_F  TM_TERM_IDF_F  TM_TERM_TF_V  TM_TERM_IDF_V
    0    term1            1              1      0.030303       1.791759
    1    term2            3              2      0.090909       1.098612
    2    term3            7              4      0.212121       0.405465

    >>> tfidf[1].head(3).collect()
         ID TM_TERMS  TM_TERM_FREQUENCY
    0  doc1    term1                  1
    1  doc1    term2                  2
    2  doc1    term3                  3

    >>> tfidf[2].head(3).collect()
          ID    CATEGORY
    0   doc1  CATEGORY_1
    1   doc2  CATEGORY_1
    2   doc3  CATEGORY_2
    """
    conn_context = data.connection_context
    if not conn_context.is_cloud_version():
        raise AttributeError("Feature not supported for on-premise.")
    lang = arg('lang', lang, str)
    param_rows = [('LANGUAGE', None, None, lang)]
    unique_id = str(uuid.uuid1()).replace('-', '_').upper()
    result_tbl = "#TM_TFIDF_RESULT_{}".format(unique_id)
    doc_term_freq = "#TM_TFIDF_DOC_TERM_FREQ_{}".format(unique_id)
    doc_category = "#TM_TFIDF_DOC_CATEGORY_{}".format(unique_id)
    output = [result_tbl, doc_term_freq, doc_category]
    try:
        call_pal_auto_with_hint(conn_context,
                                None,
                                'PAL_TF_ANALYSIS',
                                data,
                                ParameterTable().with_data(param_rows),
                                *output)
    except dbapi.Error as db_err:
        logger.exception(str(db_err))
        try_drop(conn_context, output)
        raise
    except pyodbc.Error as db_err:
        logger.exception(str(db_err.args[1]))
        try_drop(conn_context, output)
        raise
    return (conn_context.table(result_tbl), conn_context.table(doc_term_freq), conn_context.table(doc_category))

def text_classification(pred_data, ref_data=None, k_nearest_neighbours=None, thread_ratio=None, lang='EN', index_name=None, created_index=None):
    """
    This function classifies (categorizes) an input document with respect to sets of categories (taxonomies).

    Parameters
    ----------
    pred_data : DataFrame
        The prediction data for classification.

        - 1st column, ID.
        - 2nd column, Document content.

    ref_data : DataFrame or a tuple of DataFrame,

        - DataFrame, reference data
            - 1st column, ID.
            - 2nd column, Document content.
            - 3rd column, Document Category.

        The ref_data could also be a tuple of DataFrame, reference TF-IDF data:
            - 1st DataFrame
                - 1st column, TM_TERM.
                - 2nd column, TF_VALUE.
                - 3rd column, IDF_VALUE.
            - 2nd DataFrame
                - 1st column, ID.
                - 2nd column, TM_TERM.
                - 3rd column, TM_TERM_FREQUENCY.
            - 3rd DataFrame
                - 1st column, ID.
                - 2nd column, Document category.

    k_nearest_neighbours : int, optional
        Number of nearest neighbors (k).

        Defaults to 1.

    thread_ratio : float, optional
        Specifies the ratio of total number of threads that can be used by this function.
        The range of this parameter is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.
        Values outside this range are ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.0.

    lang : str, optional
        Specify the language type. Cloud HANA instance currently supports 'EN' and 'DE'.

        Defaults to 'EN'.

    index_name : str, optional
        Only for on-premise HANA instance, specify the index name. If None, it will be generated.

    created_index : {"index": xxx, "schema": xxx, "table": xxx}, optional
        Only for on-premise HANA instance. Used the created index on the given table.

    Returns
    -------
    DataFrame (Cloud)
        Text classification result, structured as follows:
            - Predict data ID.
            - TARGET.

        Statistics table, structured as follows:
            - Predict data ID.
            - K.
            - Training data ID.
            - Distance.

    DataFrame (On-Premise)
        Text classification result, structured as follows:
            - Predict data ID.
            - RANK.
            - CATEGORY_SCHEMA.
            - CATEGORY_TABLE.
            - CATEGORY_COLUMN.
            - CATEGORY_VALUE.
            - NEIGHBOR_COUNT.
            - SCORE.


    Examples
    --------

    The input DataFrame df:

    >>> df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    Invoke text_classification:

    >>> res = text_classification(df.select(df.columns[0], df.columns[1]), df)

    Result on a SAP HANA Cloud instance:

    >>> res[0].head(1).collect()
           ID     TARGET
    0    doc1 CATEGORY_1

    Result on a SAP HANA On-Premise instance:

    >>> res[0].head(1).collect()
         ID RANK  CATEGORY_SCHEMA                   CATEGORY_TABLE    CATEGORY_COLUMN  CATEGORY_VALUE  NEIGHBOR_COUNT
    0  doc1    1       "PAL_USER" "TM_CATEGORIZE_KNN_DT_6_REF_TBL"         "CATEGORY"      CATEGORY_1               1
    ...                               SCORE
    ...0.5807794005266924131092309835366905

    """
    conn_context = pred_data.connection_context
    if conn_context.is_cloud_version():
        tf_analysis_result = None
        doc_term_freq = None
        doc_category = None
        if isinstance(ref_data, DataFrame):
            tf_analysis_result, doc_term_freq, doc_category = tf_analysis(ref_data, lang)
        else:
            tf_analysis_result, doc_term_freq, doc_category = ref_data
        k_nearest_neighbours = arg('k_nearest_neighbours', k_nearest_neighbours, int)
        thread_ratio = arg('thread_ratio', thread_ratio, float)
        param_rows = [('THREAD_RATIO', None, thread_ratio, None),
                      ('K_NEAREST_NEIGHBOURS', k_nearest_neighbours, None, None)
                     ]
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#TM_TEXT_CLASSIFICATION_RESULT_{}".format(unique_id)
        stats_tbl = "#TM_TEXT_CLASSIFICATION_STATS_{}".format(unique_id)
        output = [result_tbl, stats_tbl]
        try:
            call_pal_auto_with_hint(conn_context,
                                    None,
                                    'PAL_TEXTCLASSIFICATION',
                                    tf_analysis_result,
                                    doc_term_freq,
                                    doc_category,
                                    pred_data,
                                    ParameterTable().with_data(param_rows),
                                    *output)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn_context, output)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn_context, output)
            raise
        return conn_context.table(result_tbl), conn_context.table(stats_tbl)
    else:
        if created_index:
            mater_tab = created_index["table"]
            if "schema" in mater_tab:
                schema = created_index["schema"]
            else:
                schema = conn_context.get_current_schema()
        else:
            mater_tab = 'TM_CATEGORIZE_KNN_{}_REF_TBL'.format(ref_data.name.replace('-', '_').upper())
            ref_data.save(mater_tab, force=True)
            logger.warning("Materized the dataframe to HANA table: %s.", mater_tab)
            schema = conn_context.get_current_schema()
            conn_context.add_primary_key(mater_tab, ref_data.columns[0], schema)
        if created_index is None:
            if index_name is None:
                index_name = "TM_CATEGORIZE_KNN_{}_INDEX".format(ref_data.name.replace('-', '_').upper())
            _try_drop_index(conn_context, index_name)
            _try_create_index(conn_context, index_name, mater_tab, ref_data.columns[1], schema, lang)
            logger.warning("Created index: %s.", index_name)
        if k_nearest_neighbours is None:
            k_nearest_neighbours = 1
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_df = None

        for row in pred_data.select(pred_data.columns[0]).collect().to_numpy().flatten():
            sel_statement = """
            SELECT '{6}' ID, * FROM TM_CATEGORIZE_KNN(
                DOCUMENT ({0})
                SEARCH NEAREST NEIGHBORS {1} "{4}"
                FROM "{5}"."{2}"
                RETURN TOP DEFAULT
                "{3}"
                FROM "{5}"."{2}")""".format(pred_data.filter(""""{cond}"='{rowid}'""".format(cond=pred_data.columns[0], rowid=row)).select(pred_data.columns[1]).select_statement,
                                            k_nearest_neighbours,
                                            mater_tab,
                                            ref_data.columns[2],
                                            ref_data.columns[1],
                                            schema,
                                            row)
            temp_df = conn_context.sql(sel_statement)
            if result_df:
                result_df = result_df.union(temp_df)
            else:
                result_df = temp_df
        return result_df

def _try_drop_index(conn_context, name):
    sql = "DROP FULLTEXT INDEX {}".format(quotename(name))
    try:
        with conn_context.connection.cursor() as cur:
            execute_logged(cur,
                           sql,
                           conn_context.sql_tracer,
                           conn_context)
    except:
        pass

def _try_create_index(conn_context, name, table, col, schema, lang):
    sql = """CREATE FULLTEXT INDEX {0} ON "{3}"."{1}"("{2}")
    TEXT ANALYSIS ON TEXT MINING ON LANGUAGE DETECTION ('{4}')
    TEXT MINING CONFIGURATION OVERLAY '<xml><property name="similarityFunction">COSINE</property></xml>';
    """.format(name, table, col, schema, lang)
    try:
        with conn_context.connection.cursor() as cur:
            execute_logged(cur,
                           sql,
                           conn_context.sql_tracer,
                           conn_context)
    except Exception as err:
        logger.error(err)
        pass
    ci_timeout = 7200
    if "OPCI_TIMEOUT" in os.environ:
        ci_timeout = int(os.environ["OPCI_TIMEOUT"])
    for _ in range(0, ci_timeout):
        time.sleep(1)
        is_in_queue = conn_context.sql("SELECT COUNT(*) FROM {}.{} WHERE INDEXING_STATUS({})='QUEUED';".format(quotename(schema),
                                                                                                               quotename(table),
                                                                                                               quotename(col))).collect().iat[0, 0]
        if is_in_queue == 0:
            break

def _get_basic_func(func, pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index):
    conn_context = pred_data.connection_context
    tf_analysis_result = None
    doc_term_freq = None
    doc_category = None
    if conn_context.is_cloud_version():
        if isinstance(ref_data, DataFrame):
            tf_analysis_result, doc_term_freq, doc_category = tf_analysis(ref_data, lang)
        else:
            tf_analysis_result, doc_term_freq, doc_category = ref_data
        top = arg('top', top, int)
        threshold = arg('threshold', threshold, float)
        param_rows = []
        if threshold is not None:
            param_rows.append(('THRESHOLD', None, threshold, None))
        if top is not None:
            param_rows.append(('TOP', top, None, None))
        if thread_ratio is not None:
            param_rows.append(('THREAD_RATIO', None, thread_ratio, None))
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#TM_{}_RESULT_{}".format(func, unique_id)
        try:
            call_pal_auto_with_hint(conn_context,
                                    None,
                                    func,
                                    tf_analysis_result,
                                    doc_term_freq,
                                    doc_category,
                                    pred_data,
                                    ParameterTable().with_data(param_rows),
                                    result_tbl)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn_context, result_tbl)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn_context, result_tbl)
            raise
        return conn_context.table(result_tbl)
    else:
        if created_index:
            mater_tab = created_index["table"]
            if "schema" in mater_tab:
                schema = created_index["schema"]
            else:
                schema = conn_context.get_current_schema()
        else:
            mater_tab = '{}_{}_REF_TBL'.format(func, ref_data.name.replace('-', '_').upper())
            ref_data.save(mater_tab, force=True)
            logger.warning("Materized the dataframe to HANA table: %s.", mater_tab)
            schema = conn_context.get_current_schema()
            conn_context.add_primary_key(mater_tab, ref_data.columns[0], schema)
        if created_index is None:
            if index_name is None:
                index_name = "{}_{}_INDEX".format(func, ref_data.name.replace('-', '_').upper())
            _try_drop_index(conn_context, index_name)
            _try_create_index(conn_context, index_name, mater_tab, ref_data.columns[1], schema, lang)
            logger.warning("Created index: %s.", index_name)
        if top is None:
            top = 'DEFAULT'
        sel_statement = None
        if func in ("TM_GET_RELATED_DOCUMENTS"):
            sel_statement = """
                            SELECT T.* FROM {0}(
                                DOCUMENT ({1})
                                SEARCH "{4}"
                                FROM "{5}"."{2}"
                                RETURN
                                TOP {3}
                                {6}) AS T""".format(func,
                                                    pred_data.select_statement,
                                                    mater_tab,
                                                    top,
                                                    ref_data.columns[1],
                                                    schema,
                                                    ref_data.columns[0])
        elif func in ("TM_GET_RELEVANT_TERMS"):
            sel_statement = """
                            SELECT * FROM {0}(
                                DOCUMENT ({1})
                                SEARCH "{4}"
                                FROM "{5}"."{2}"
                                RETURN
                                TOP {3})""".format(func, pred_data.select_statement, mater_tab, top, ref_data.columns[1], schema)
        elif func in ("TM_GET_RELEVANT_DOCUMENTS"):
            sel_statement = """
                            SELECT T.* FROM {0}(
                                TERM '{1}'
                                SEARCH "{4}"
                                FROM "{5}"."{2}"
                                RETURN
                                TOP {3}
                                {6}) AS T""".format(func,
                                                    pred_data.collect().iat[0, 0],
                                                    mater_tab,
                                                    top,
                                                    ref_data.columns[1],
                                                    schema,
                                                    ref_data.columns[0])
        else:
            sel_statement = """
                            SELECT * FROM {0}(
                                TERM '{1}'
                                SEARCH "{4}"
                                FROM "{5}"."{2}"
                                RETURN
                                TOP {3})""".format(func, pred_data.collect().iat[0, 0], mater_tab, top, ref_data.columns[1], schema)
        return conn_context.sql(sel_statement)

def get_related_doc(pred_data, ref_data=None, top=None, threshold=None, lang='EN', index_name=None, thread_ratio=None, created_index=None):
    """
    This function returns the top-ranked related documents for a query document based on Term Frequency - Inverse Document Frequency(TF-IDF) result or reference data..

    Parameters
    ----------
    pred_data : DataFrame

        - 1st column, Document content.

    ref_data : DataFrame or a tuple of DataFrame,

        - DataFrame, reference data
            - 1st column, ID
            - 2nd column, Document content
            - 3rd column, Document Category

        The ref_data could also be a tuple of DataFrame, reference TF-IDF data:
            - 1st DataFrame, TF-IDF Result
                - 1st column, TM_TERM
                - 2nd column, TF_VALUE
                - 3rd column, IDF_VALUE
            - 2nd DataFrame, Doc Term Freq Table
                - 1st column, ID
                - 2nd column, TM_TERM
                - 3rd column, TM_TERM_FREQUENCY
            - 3rd DataFrame, Doc Category Table
                - 1st column, ID
                - 2nd column, Document category

    top : int, optional
        Only show top N results. If 0, it shows all.

        Defaults to 0.

    threshold : float, optional
        Only the results which score bigger than this value will be put into the result table.

        Defaults to 0.0.

    lang : str, optional
        Specify the language type. Cloud HANA instance currently supports 'EN' and 'DE'.

        Defaults to 'EN'.

    index_name : str, optional
        Only for on-premise SAP HANA instance, specify the index name.

    thread_ratio : float, optional
        Only for cloud version.
        Specifies the ratio of total number of threads that can be used by this function.
        The range of this parameter is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.
        Values outside this range are ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.0.

    created_index : {"index": xxx, "schema": xxx, "table": xxx}, optional
        Only for on-premise HANA instance. Used the created index on the given table.

    Returns
    -------
    DataFrame


    Examples
    --------

    The input DataFrame ref_df:

    >>> ref_df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    The input DataFrame pred_df:

    >>> pred_df.collect()
                       CONTENT
    0  term2 term2 term3 term3

    Invoke the function on a SAP HANA Cloud instance:
    tfidf is a DataFrame returned by tf_analysis function, please refer to the examples section of tf_analysis for its content.

    >>> get_related_doc(pred_df, tfidf).collect()
           ID       SCORE
    0    doc2    0.891550
    1    doc1    0.804670
    2    doc3    0.042024
    3    doc4    0.021225

    Invoke the function on a SAP HANA On-Premise instance:

    >>> res = get_related_doc(df_test1_onpremise, df_onpremise)
    >>> res.collect()
       ID    RANK   TOTAL_TERM_COUNT  TERM_COUNT  CORRELATIONS  FACTORS  ROTATED_FACTORS  CLUSTER_LEVEL  CLUSTER_LEFT
    0  doc2     1                  6           3          None     None             None           None          None
    1  doc1     2                  6           3          None     None             None           None          None
    2  doc3     3                  6           3          None     None             None           None          None
    3  doc4     4                  9           3          None     None             None           None          None
    ... CLUSTER_RIGHT  HIGHLIGHTED_DOCUMENT  HIGHLIGHTED_TERMTYPES                                   SCORE
    ...          None                  None                   None    0.8915504731053067732915451415465213
    ...          None                  None                   None    0.8046698732333942283290184604993556
    ...          None                  None                   None   0.04202449735779462125506711345224176
    ...          None                  None                   None   0.02122540837399113089478674964993843

 """
    if pred_data.connection_context.is_cloud_version():
        return _get_basic_func("PAL_TMGETRELATEDDOC", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)
    else:
        return _get_basic_func("TM_GET_RELATED_DOCUMENTS", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)

def get_related_term(pred_data, ref_data=None, top=None, threshold=None, lang='EN', index_name=None, thread_ratio=None, created_index=None):
    """
    This function returns the top-ranked related terms for a query term based on Term Frequency - Inverse Document Frequency(TF-IDF) result or reference data.

    Parameters
    ----------
    pred_data : DataFrame

        - 1st column, Document content.

    ref_data : DataFrame or a tuple of DataFrame,

        - DataFrame, reference data
            - 1st column, ID.
            - 2nd column, Document content.
            - 3rd column, Document Category.

        The ref_data could also be a tuple of DataFrame, reference TF-IDF data:
            - 1st DataFrame
                - 1st column, TM_TERM.
                - 2nd column, TF_VALUE.
                - 3rd column, IDF_VALUE.
            - 2nd DataFrame
                - 1st column, ID.
                - 2nd column, TM_TERM.
                - 3rd column, TM_TERM_FREQUENCY.
            - 3rd DataFrame
                - 1st column, ID.
                - 2nd column, Document category.

    top : int, optional
        Show top N results. If 0, it shows all.

        Defaults to 0.

    threshold : float, optional
        Only the results which score bigger than this value will be put into a result table.

        Defaults to 0.0.

    lang : str, optional
        Specify the language type. Cloud HANA instance currently supports 'EN' and 'DE'.

        Defaults to 'EN'.

    index_name : str, optional
        Only for on-premise HANA isntance, specify the index name.

    thread_ratio : float, optional
        Only for cloud version.
        Specifies the ratio of total number of threads that can be used by this function.
        The range of this parameter is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.
        Values outside this range are ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.0.

    created_index : {"index": xxx, "schema": xxx, "table": xxx}, optional
        Only for on-premise HANA instance. Used the created index on the given table.

    Returns
    -------
    DataFrame

    Examples
    --------

    The input DataFrame ref_df:

    >>> ref_df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    The input DataFrame pred_df:

    >>> pred_df.collect()
      CONTENT
    0   term3

    Invoke the function on a SAP HANA Cloud instance,

    >>> get_related_term(pred_df, ref_df).collect()
            ID       SCORE
    0    term3    1.000000
    1    term2    0.923760
    2    term1    0.774597
    3    term4    0.550179
    4    term5    0.346410

    Invoke the function on a SAP HANA On-Premise instance:

    >>> res = get_related_term(pred_df, ref_df)
    >>> res.collect()
      RANK  TERM  NORMALIZED_TERM  TERM_TYPE  TERM_FREQUENCY  DOCUMENT_FREQUENCY  CORRELATIONS
    0    1 term3            term3       noun               7                   4          None
    1    2 term2            term2       noun               3                   2          None
    2    3 term1            term1       noun               1                   1          None
    3    4 term4            term4       noun               9                   5          None
    4    5 term5            term5       noun               9                   2          None
    ... FACTORS  ROTATED_FACTORS  CLUSTER_LEVEL  CLUSTER_LEFT  CLUSTER_RIGHT                                 SCORE
    ...    None             None           None          None           None  1.0000003613794823387195265240734440
    ...    None             None           None          None           None  0.9237607645314674931213971831311937
    ...    None             None           None          None           None  0.7745969491648266869177064108953346
    ...    None             None           None          None           None  0.5501794128048571597133786781341769
    ...    None             None           None          None           None  0.3464102866993003515538873671175679
    """

    if pred_data.connection_context.is_cloud_version():
        return _get_basic_func("PAL_TMGETRELATEDTERM", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)
    else:
        return _get_basic_func("TM_GET_RELATED_TERMS", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)

def get_relevant_doc(pred_data, ref_data=None, top=None, threshold=None, lang='EN', index_name=None, thread_ratio=None, created_index=None):

    """
    This function returns the top-ranked documents that are relevant to a term based on Term Frequency - Inverse Document Frequency(TF-IDF) result or reference data.

    Parameters
    ----------
    pred_data : DataFrame

        - 1st column, Document content.

    ref_data : DataFrame or a tuple of DataFrame,

        - DataFrame, reference data
            - 1st column, ID.
            - 2nd column, Document content.
            - 3rd column, Document Category.

        The ref_data could also be a tuple of DataFrame, reference TF-IDF data:
            - 1st DataFrame
                - 1st column, TM_TERM.
                - 2nd column, TF_VALUE.
                - 3rd column, IDF_VALUE.
            - 2nd DataFrame
                - 1st column, ID.
                - 2nd column, TM_TERM.
                - 3rd column, TM_TERM_FREQUENCY.
            - 3rd DataFrame
                - 1st column, ID.
                - 2nd column, Document category.

    top : int, optional
        Show top N results. If 0, it shows all.

        Defaults to 0.

    threshold : float, optional
        Only the results which score bigger than this value will be put into a result table.

        Defaults to 0.0.

    lang : str, optional
        Specify the language type. Cloud HANA instance currently supports 'EN' and 'DE'.

        Defaults to 'EN'.

    index_name : str, optional
        Only for on-premise HANA isntance, specify the index name.

    thread_ratio : float, optional
        Only for cloud version.
        Specifies the ratio of total number of threads that can be used by this function.
        The range of this parameter is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.
        Values outside this range are ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.0.

    created_index : {"index": xxx, "schema": xxx, "table": xxx}, optional
        Only for on-premise HANA instance. Used the created index on the given table.

    Returns
    -------
    DataFrame

    Examples
    --------

    The input DataFrame ref_df:

    >>> ref_df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    The input DataFrame pred_df:

    >>> pred_df.collect()
                       CONTENT
    0                    term3

    Invoke the function on a SAP HANA Cloud instance:

    >>> get_relevant_doc(pred_df, ref_df).collect()
           ID       SCORE
    0    doc1    0.774597
    1    doc2    0.516398
    2    doc3    0.258199
    3    doc4    0.258199

    Invoke the function on a SAP HANA On-Premise instance:

    >>> res = get_relevant_doc(pred_data, ref_data, top=4)
    >>> res.collect()
         ID    RANK   TOTAL_TERM_COUNT  TERM_COUNT  CORRELATIONS  FACTORS  ROTATED_FACTORS  CLUSTER_LEVEL  CLUSTER_LEFT
    0  doc1       1                  6           3          None     None             None           None          None
    1  doc2       2                  6           3          None     None             None           None          None
    2  doc3       3                  6           3          None     None             None           None          None
    3  doc4       4                  9           3          None     None             None           None          None
    ... CLUSTER_RIGHT  HIGHLIGHTED_DOCUMENT  HIGHLIGHTED_TERMTYPES                                   SCORE
    ...          None                  None                   None    0.7745969491648266869177064108953346
    ...          None                  None                   None    0.5163979661098845319600059156073257
    ...          None                  None                   None    0.2581989830549422659800029578036629
    ...          None                  None                   None    0.2581989830549422659800029578036629

    """
    if pred_data.connection_context.is_cloud_version():
        return _get_basic_func("PAL_TMGETRELEVANTDOC", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)
    else:
        return _get_basic_func("TM_GET_RELEVANT_DOCUMENTS", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)

def get_relevant_term(pred_data, ref_data=None, top=None, threshold=None, lang='EN', index_name=None, thread_ratio=None, created_index=None):
    """
    This function returns the top-ranked relevant terms that describe a document based on Term Frequency - Inverse Document Frequency(TF-IDF) result or reference data.

    Parameters
    ----------
    pred_data : DataFrame

        - 1st column, Document content.

    ref_data : DataFrame or a tuple of DataFrame,

        - DataFrame, reference data
            - 1st column, ID.
            - 2nd column, Document content.
            - 3rd column, Document Category.

        The ref_data could also be a tuple of DataFrame, reference TF-IDF data:
            - 1st DataFrame
                - 1st column, TM_TERM.
                - 2nd column, TF_VALUE.
                - 3rd column, IDF_VALUE.
            - 2nd DataFrame
                - 1st column, ID.
                - 2nd column, TM_TERM.
                - 3rd column, TM_TERM_FREQUENCY.
            - 3rd DataFrame
                - 1st column, ID.
                - 2nd column, Document category.

    top : int, optional
        Show top N results. If 0, it shows all.

        Defaults to 0.

    threshold : float, optional
        Only the results which score bigger than THRESHOLD will be put into a result table.

        Defaults to 0.0.

    lang : str, optional
        Specify the language type. Cloud HANA instance currently supports 'EN' and 'DE'.

        Defaults to 'EN'.

    index_name : str, optional
        Only for on-premise HANA isntance, specify the index name.

    thread_ratio : float, optional
        Only for cloud version.
        Specifies the ratio of total number of threads that can be used by this function.
        The range of this parameter is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.
        Values outside this range are ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.0.

    created_index : {"index": xxx, "schema": xxx, "table": xxx}, optional
        Only for on-premise HANA instance. Used the created index on the given table.

    Returns
    -------
    DataFrame

    Examples
    --------

    The input DataFrame ref_df:

    >>> ref_df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    The input DataFrame pred_df:

    >>> pred_df.collect()
      CONTENT
    0   term3

    Invoke the function on a SAP HANA Cloud instance,

    >>> get_relevant_term(pred_df, ref_df).collect()
            ID   SCORE
    0    term3     1.0

    Invoke the function on a SAP HANA On-Premise instance:

    >>> res = get_relevant_term(pred_df, ref_df)
    >>> res.collect()
      RANK  TERM  NORMALIZED_TERM  TERM_TYPE  TERM_FREQUENCY  DOCUMENT_FREQUENCY  CORRELATIONS
    0    1 term3            term3       noun               7                   4          None
    ... FACTORS  ROTATED_FACTORS  CLUSTER_LEVEL  CLUSTER_LEFT  CLUSTER_RIGHT                                 SCORE
    ...    None             None           None          None           None   1.000002901113076436701021521002986

    """
    if pred_data.connection_context.is_cloud_version():
        return _get_basic_func("PAL_TMGETRELEVANTTERM", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)
    else:
        return _get_basic_func("TM_GET_RELEVANT_TERMS", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)

def get_suggested_term(pred_data, ref_data=None, top=None, threshold=None, lang='EN', index_name=None, thread_ratio=None, created_index=None):
    """
    This function returns the top-ranked terms that match an initial substring based on Term Frequency - Inverse Document Frequency(TF-IDF) result or reference data.

    Parameters
    ----------
    pred_data : DataFrame

        - 1st column, Document content.

    ref_data : DataFrame or a tuple of DataFrame,

        - DataFrame, reference data
            - 1st column, ID.
            - 2nd column, Document content.
            - 3rd column, Document Category.

        The ref_data could also be a tuple of DataFrame, reference TF-IDF data:
            - 1st DataFrame
                - 1st column, TM_TERM.
                - 2nd column, TF_VALUE.
                - 3rd column, IDF_VALUE.
            - 2nd DataFrame
                - 1st column, ID.
                - 2nd column, TM_TERM.
                - 3rd column, TM_TERM_FREQUENCY.
            - 3rd DataFrame
                - 1st column, ID.
                - 2nd column, Document category.

    top : int, optional
        Show top N results. If 0, it shows all.

        Defaults to 0.

    threshold : float, optional
        Only the results which score bigger than this value will be put into a result table.

        Defaults to 0.0.

    lang : str, optional
        Specify the language type. Cloud HANA instance currently supports 'EN' and 'DE'.

        Defaults to 'EN'.

    index_name : str, optional
        Only for on-premise HANA isntance, specify the index name.

    thread_ratio : float, optional
        Only for cloud version.
        Specifies the ratio of total number of threads that can be used by this function.
        The range of this parameter is from 0 to 1, where 0 means only using 1 thread,
        and 1 means using at most all the currently available threads.
        Values outside this range are ignored and this function heuristically determines the number of threads to use.

        Defaults to 0.0.

    created_index : {"index": xxx, "schema": xxx, "table": xxx}, optional
        Only for on-premise HANA instance. Used the created index on the given table.

    Returns
    -------
    DataFrame

    Examples
    --------

    The input DataFrame ref_df:

    >>> ref_df.collect()
          ID                                                  CONTENT       CATEGORY
    0   doc1                      term1 term2 term2 term3 term3 term3     CATEGORY_1
    1   doc2                      term2 term3 term3 term4 term4 term4     CATEGORY_1
    2   doc3                      term3 term4 term4 term5 term5 term5     CATEGORY_2
    3   doc4    term3 term4 term4 term5 term5 term5 term5 term5 term5     CATEGORY_2
    4   doc5                                              term4 term6     CATEGORY_3
    5   doc6                                  term4 term6 term6 term6     CATEGORY_3

    The input DataFrame pred_df:

    >>> pred_df.collect()
      CONTENT
    0   term3

    Invoke the function on a SAP HANA Cloud instance,

    >>> get_suggested_term(pred_df, ref_df).collect()
            ID     SCORE
    0    term3       1.0

    Invoke the function on a SAP HANA On-Premise instance:

    >>> res = get_suggested_term(pred_df, ref_df)
    >>> res.collect()
      RANK   TERM  NORMALIZED_TERM  TERM_TYPE  TERM_FREQUENCY  DOCUMENT_FREQUENCY                                SCORE
    0    1  term3            term3       noun               7                   4  0.999999999999999888977697537484346
    """
    if pred_data.connection_context.is_cloud_version():
        return _get_basic_func("PAL_TMGETSUGGESTEDTERM", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)
    else:
        return _get_basic_func("TM_GET_SUGGESTED_TERMS", pred_data, ref_data, top, threshold, lang, index_name, thread_ratio, created_index)

class TFIDF(PALBase):  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    r"""
    Class for term frequency–inverse document frequency.

    Parameters
	----------

    Examples
    --------
    Input dataframe for analysis:

    >>> df_train.collect()
            ID      CONTENT
        0   doc1    term1 term2 term2 term3 term3 term3
        1   doc2    term2 term3 term3 term4 term4 term4
        2   doc3    term3 term4 term4 term5 term5 term5
        3   doc5    term3 term4 term4 term5 term5 term5 term5 term5 term5
        4   doc4    term4 term6
        5   doc6    term4 term6 term6 term6

    Creating TFIDF instance:

    >>> tfidf = TFIDF()

    Performing text_collector() on given dataframe:

    >>> idf, _ = tfidf.text_collector(data=self.df_train)

    >>> idf.collect()
            TM_TERMS    TM_TERM_IDF_VALUE
        0   term1       1.791759
        1   term2       1.098612
        2   term3       0.405465
        3   term4       0.182322
        4   term5       1.098612
        5   term6       1.098612

    Performing text_tfidf() on given dataframe:

    >>> result = tfidf.text_tfidf(data=self.df_train)

    >>> result.collect()
            ID      TERMS   TF_VALUE    TFIDF_VALUE
        0   doc1    term1   1.0         1.791759
        1   doc1    term2   2.0         2.197225
        2   doc1    term3   3.0         1.216395
        3   doc2    term2   1.0         1.098612
        4   doc2    term3   2.0         0.810930
        5   doc2    term4   3.0         0.546965
        6   doc3    term3   1.0         0.405465
        7   doc3    term4   2.0         0.364643
        8   doc3    term5   3.0         3.295837
        9   doc5    term3   1.0         0.405465
        10  doc5    term4   2.0         0.364643
        11  doc5    term5   6.0         6.591674
        12  doc4    term4   1.0         0.182322
        13  doc4    term6   1.0         1.098612
        14  doc6    term4   1.0         0.182322
        15  doc6    term6   3.0         3.295837
    """

    def __init__(self):
        super(TFIDF, self).__init__()
        self.idf = None
        self.extend = None

    def text_collector(self, data):
        """
        Its use is primarily compute inverse document frequency of documents which provided by user.

        Parameters
        ----------
        data : DataFrame
            Data to be analysis.
            The first column of the input data table is assumed to be an ID column.

        Returns
        -------
        DataFrame
            - Inverse document frequency of documents.
            - Extended table.
        """
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        tables = ['TERM-IDF', 'EXTEND_OUT']
        tables = ["#PAL_IDF_{}_TBL_{}_{}".format(
            tbl, self.id, unique_id) for tbl in tables]
        idf_tbl, extend_tbl = tables
        try:
            self._call_pal_auto(conn,
                                "PAL_TEXT_COLLECT",
                                data,
                                ParameterTable(),
                                *tables)
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(conn, tables)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(conn, tables)
            raise
        self.idf = conn.table(idf_tbl)
        return conn.table(idf_tbl), conn.table(extend_tbl)

    def text_tfidf(self, data, idf=None):
        """
        Its use is primarily compute term frequency - inverse document frequency by document.

        Parameters
        ----------
        data : DataFrame
            Data to be analysis.

            The first column of the input data table is assumed to be an ID column.
        idf : DataFrame, optional
            Inverse document frequency of documents.

        Returns
        -------
        DataFrame
            - Term frequency - inverse document frequency by document.
        """
        if not idf:
            if not self.idf:
                msg = "text_collector() has not been excucated."
                logger.error(msg)
                raise ValueError(msg)
            idf = self.idf
        conn = data.connection_context
        require_pal_usable(conn)
        unique_id = str(uuid.uuid1()).replace('-', '_').upper()
        result_tbl = "#PAL_TFIDF_RESULT_{}_{}".format(self.id, unique_id)
        try:
            self._call_pal_auto(conn,
                                "PAL_TEXT_TFIDF",
                                data,
                                idf,
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
