"""
PAL-specific helper functionality.
"""
import logging
import uuid
import inspect
import json
import re
import time
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
import hana_ml.ml_base
import hana_ml.ml_exceptions

from hana_ml.dataframe import quotename, DataFrame
from hana_ml.algorithms.pal import sqlgen

from hana_ml.model_storage_services import ModelSavingServices

# Expose most contents of ml_base in pal_base for import convenience.
# pylint: disable=unused-import
# pylint: disable=attribute-defined-outside-init
# pylint: disable=bare-except
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=line-too-long
# pylint: disable=super-with-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=superfluous-parens
# pylint: disable=anomalous-backslash-in-string
# pylint: disable=too-many-public-methods
# pylint: disable=consider-using-f-string, raising-bad-type
# pylint: disable=broad-except
from hana_ml.ml_base import (
    Table,
    INTEGER,
    DOUBLE,
    NVARCHAR,
    NCLOB,
    arg,
    create,
    materialize,
    try_drop,
    parse_one_dtype,
    execute_logged,
    logged_without_execute,
    colspec_from_df,
    ListOfStrings,
    ListOfTuples,
    TupleOfIntegers,
    _TEXT_TYPES,
    _INT_TYPES,
)
from hana_ml.algorithms.pal.sqlgen import ParameterTable

logger = logging.getLogger(__name__)

MINIMUM_HANA_VERSION_PREFIX = '2.00.030'

_SELECT_HANA_VERSION = ("SELECT VALUE FROM SYS.M_SYSTEM_OVERVIEW " +
                        "WHERE NAME='Version'")
_SELECT_PAL = "SELECT * FROM SYS.AFL_PACKAGES WHERE PACKAGE_NAME='PAL'"
_SELECT_PAL_PRIVILEGE = (
    "SELECT * FROM SYS.EFFECTIVE_ROLES " +
    "WHERE USER_NAME=CURRENT_USER AND " +
    "ROLE_SCHEMA_NAME IS NULL AND "
    "ROLE_NAME IN ('AFL__SYS_AFL_AFLPAL_EXECUTE', " +
    "'AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION')"
)

def pal_param_register():
    """
    Register PAL parameters after PAL object has been initialized.
    """
    frame = inspect.currentframe()
    params = frame.f_back.f_locals
    try:
        params.pop('self')
    except KeyError:
        pass
    try:
        params.pop('functionality')
    except KeyError:
        pass
    try:
        params.pop('data')
    except KeyError:
        pass
    serializable_params = {}
    for param_key, param_value in params.items():
        try:
            json.dumps(param_value)
            serializable_params[param_key] = param_value
        except Exception as err:
            logger.warning(err)
            pass
    return serializable_params

class PALBase(hana_ml.ml_base.MLBase, ModelSavingServices):
    """
    Subclass for PAL-specific functionality.
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, conn_context=None):
        super(PALBase, self).__init__(conn_context)
        ModelSavingServices.__init__(self)
        self.execute_statement = None
        self.with_hint = None
        self.with_hint_anonymous_block = False
        self._fit_param = None
        self._predict_param = None
        self._score_param = None
        self._fit_call = None
        self._predict_call = None
        self._score_call = None
        self._fit_anonymous_block = None
        self._predict_anonymous_block = None
        self._score_anonymous_block = None
        self._fit_output_table_names = None
        self._predict_output_table_names = None
        self._score_output_table_names = None
        self._fit_args = None
        self._predict_args = None
        self._score_args = None
        self.runtime = None
        self.base_fit_proc_name = None
        self.base_predict_proc_name = None
        self._convert_bigint = False

    def apply_with_hint(self, with_hint, apply_to_anonymous_block=False):
        """

        Parameters
        ----------
        with_hint : str
            The hint clauses.
        apply_to_anonymous_block : bool, optional
            If True, it will be applied to the anonymous block.
        """
        self.with_hint = with_hint
        if apply_to_anonymous_block:
            self.with_hint = "*{}".format(self.with_hint)

    def enable_parallel_by_parameter_partitions(self, apply_to_anonymous_block=False):
        """
        Enable parallel by parameter partitions.

        Parameters
        ----------
        apply_to_anonymous_block : bool, optional
            If True, it will be applied to the anonymous block.
        """
        self.with_hint = 'PARALLEL_BY_PARAMETER_PARTITIONS(p1)'
        if apply_to_anonymous_block:
            self.with_hint = "*{}".format(self.with_hint)

    def enable_no_inline(self, apply_to_anonymous_block=False):
        """
        Enable no inline.

        Parameters
        ----------
        apply_to_anonymous_block : bool, optional
            If True, it will be applied to the anonymous block.
        """
        self.with_hint = 'no_inline'
        if apply_to_anonymous_block:
            self.with_hint = "*{}".format(self.with_hint)

    def disable_with_hint(self):
        """
        Disable with hint.
        """
        self.with_hint = None

    def enable_convert_bigint(self):
        """
        Allow the conversion from bigint to double.

        Defaults to True.
        """
        self._convert_bigint = True

    def disable_convert_bigint(self):
        """
        Disable the bigint conversion
        """
        self._convert_bigint = False

    @property
    def fit_hdbprocedure(self):
        """
        Return the generated hdbprocedure for fit.
        """
        if self._fit_call is None:
            raise("Please run fit function first!")
        proc_name = "HANAMLAPI_BASE_{}_TRAIN".format(re.findall("_SYS_AFL.(.+)\(", self._fit_call)[0])
        self.base_fit_proc_name = proc_name
        inputs = []
        outputs = []
        count = 0
        conn = None
        for inp in self._fit_args:
            if isinstance(inp, DataFrame):
                conn = inp.connection_context
                input_tt = []
                for key, val in inp.get_table_structure().items():
                    input_tt.append("{} {}".format(quotename(key), val))
                inputs.append("in in_{} TABLE({})".format(count,
                                                          ", ".join(input_tt)))
                count = count + 1
        count = 0
        for output in self._fit_output_table_names:
            output_tt = []
            for key, val in conn.table(output).get_table_structure().items():
                output_tt.append("{} {}".format(quotename(key), val))
            outputs.append("out out_{} TABLE({})".format(count,
                                                         ", ".join(output_tt)))
            count = count + 1
        proc_header = "PROCEDURE {}(\n{})\nLANGUAGE SQLSCRIPT\nSQL SECURITY INVOKER\nAS\nBEGIN\n".format(proc_name.lower(),\
            ",\n".join(inputs + outputs))
        body = re.search(r'DECLARE [\s\S]+UNNEST\(:param_name, :int_value, :double_value, :string_value\);', self._fit_anonymous_block).group(0)
        return proc_header + body + "\nCALL " + self._fit_call + "\nEND"

    @property
    def predict_hdbprocedure(self):
        """
        Return the generated hdbprocedure for predict.
        """
        if self._predict_call is None:
            raise("Please run predict function first!")
        proc_name = "HANAMLAPI_BASE_{}_APPLY".format(re.findall("_SYS_AFL.(.+)\(", self._predict_call)[0])
        self.base_predict_proc_name = proc_name
        inputs = []
        outputs = []
        count = 0
        conn = None
        for inp in self._predict_args:
            if isinstance(inp, DataFrame):
                conn = inp.connection_context
                input_tt = []
                for key, val in inp.get_table_structure().items():
                    input_tt.append("{} {}".format(key, val))
                inputs.append("IN in_{} TABLE({})".format(count,
                                                          ", ".join(input_tt)))
                count = count + 1
        count = 0
        for output in self._predict_output_table_names:
            output_tt = []
            for key, val in conn.table(output).get_table_structure().items():
                output_tt.append("{} {}".format(key, val))
            outputs.append("OUT out_{} TABLE({})".format(count,
                                                         ", ".join(output_tt)))
            count = count + 1
        proc_header = "PROCEDURE {}(\n{})\nLANGUAGE SQLSCRIPT\nSQL SECURITY INVOKER\nAS\nBEGIN\n".format(proc_name,\
            ",\n".join(inputs + outputs))
        body = re.search(r'DECLARE [\s\S]+UNNEST\(:param_name, :int_value, :double_value, :string_value\);', self._predict_anonymous_block).group(0)
        return proc_header + body + "\nCALL " + self._predict_call + "\nEND"

    def consume_fit_hdbprocedure(self, proc_name, in_tables=None, out_tables=None):
        """
        Return the generated consume hdbprocedure for fit.
        """
        result = {}
        result["base"] = self.fit_hdbprocedure
        proc_header = "PROCEDURE {}()\nLANGUAGE SQLSCRIPT\nSQL SECURITY INVOKER\nAS\nBEGIN\n".format(proc_name)
        in_vars = []
        call_in_vars = []
        if in_tables:
            for seq, in_var in enumerate(in_tables):
                in_vars.append("in_{} = SELECT * FROM {};".format(seq, quotename(in_var)))
                call_in_vars.append(":in_{}".format(seq))
        body = "\n".join(in_vars) + "\n"
        call_out_vars = []
        outputs = []
        if out_tables:
            for seq, out_var in enumerate(out_tables):
                call_out_vars.append("out_{}".format(seq))
                outputs.append("TRUNCATE TABLE {0};\nINSERT INTO {0} SELECT * FROM :{1};".format(quotename(out_var),
                                                                                                "out_{}".format(seq)))
        body = body + "CALL {} ({});".format(self.base_fit_proc_name,
                                             re.findall("\((.+)\)", self._fit_call)[0].replace(":params, ", ""))
        result["consume"] = proc_header + body + "\n" + "\n".join(outputs) + "\nEND"
        return result

    def consume_predict_hdbprocedure(self, proc_name, in_tables=None, out_tables=None):
        """
        Return the generated consume hdbprocedure for predict.
        """
        result = {}
        result["base"] = self.predict_hdbprocedure
        proc_header = "PROCEDURE {}()\nLANGUAGE SQLSCRIPT\nSQL SECURITY INVOKER\nAS\nBEGIN\n".format(proc_name)
        in_vars = []
        call_in_vars = []
        if in_tables:
            for seq, in_var in enumerate(in_tables):
                in_vars.append("in_{} = SELECT * FROM {};".format(seq, quotename(in_var)))
                call_in_vars.append(":in_{}".format(seq))
        body = "\n".join(in_vars) + "\n"
        call_out_vars = []
        outputs = []
        if out_tables:
            for seq, out_var in enumerate(out_tables):
                call_out_vars.append("out_{}".format(seq))
                outputs.append("TRUNCATE TABLE {0};\nINSERT INTO {0} SELECT * FROM :{1};".format(quotename(out_var),
                                                                                                "out_{}".format(seq)))
        body = body + "CALL {} ({});".format(self.base_predict_proc_name,
                                             re.findall("\((.+)\)", self._predict_call)[0].replace(":params, ", ""))
        result["consume"] = proc_header + body + "\n" + "\n".join(outputs) + "\nEND"
        return result


    def set_model(self, model):
        """
        set model.

        Parameters
        ----------
        model : DataFrame
            The model DataFrame to be loaded.
        """
        self.model_ = model

    def set_scale_out(self,
                      route_to=None,
                      no_route_to=None,
                      route_by=None,
                      route_by_cardinality=None,
                      data_transfer_cost=None,
                      route_optimization_level=None,
                      workload_class=None,
                      apply_to_anonymous_block=False):
        """
        HANA statement routing.

        Parameters
        ----------
        route_to : str, optional
            Routes the query to the specified volume ID or service type.
        no_route_to : str or list of str, optional
            Avoids query routing to a specified volume ID or service type.
        route_by : str, optional
            Routes the query to the hosts related to the base table(s) of the specified projection view(s).
        route_by_cardinality : str or list of str, optional
            Routes the query to the hosts related to the base table(s) of the specified projection view(s) with the highest cardinality from the input list.
        data_transfer_cost : int, optional
            Guides the optimizer to use the weighting factor for the data transfer cost. The value 0 ignores the data transfer cost.
        route_optimization_level : {'mininal', 'all'}, optional
            Guides the optimizer to compile with ROUTE_OPTIMIZATION_LEVEL (MINIMAL) or to default to ROUTE_OPTIMIZATION_LEVEL.
            If the MINIMAL compiled plan is cached, then it compiles once more using the default optimization level during the first execution.
            This hint is primarily used to shorten statement routing decisions during the initial compilation.
        workload_class : str, optional
            Routes the query via workload class. ``route_to`` statement hint has higher precedence than ``workload_class`` statement hint.

        apply_to_anonymous_block : bool, optional
            If True it will be applied to the anonymous block.
        """
        hint_str = []
        if route_to:
            hint_str.append('ROUTE_TO({})'.format(route_to))
        if no_route_to:
            if isinstance(no_route_to, (list, tuple)):
                no_route_to = list(map(str, no_route_to))
                no_route_to = ", ".join(no_route_to)
            hint_str.append('NO_ROUTE_TO({})'.format(no_route_to))
        if route_by:
            hint_str.append('ROUTE_BY({})'.format(route_by))
        if route_by_cardinality:
            if isinstance(route_by_cardinality, (list, tuple)):
                route_by_cardinality = list(map(str, route_by_cardinality))
                route_by_cardinality = ", ".join(route_by_cardinality)
            hint_str.append('ROUTE_BY_CARDINALITY({})'.format(route_by_cardinality))
        if data_transfer_cost:
            hint_str.append('DATA_TRANSFER_COST({})'.format(str(data_transfer_cost)))
        if route_optimization_level:
            self._arg('route_optimization_level', route_optimization_level.upper(), {'minimal': 'MINIMAL', 'all': 'ALL'})
            hint_str.append('ROUTE_OPTIMIZATION_LEVEL({})'.format(route_optimization_level.upper()))
        if workload_class:
            hint_str.append('WORKLOAD_CLASS({})'.format(quotename(workload_class)))
        if len(hint_str) > 0:
            self.with_hint = ", ".join(hint_str)
        if apply_to_anonymous_block:
            self.with_hint = "*{}".format(self.with_hint)

    def get_fit_execute_statement(self):
        """
        Return the execute_statement for training.
        """
        return self._fit_anonymous_block

    def get_predict_execute_statement(self):
        """
        Return the execute_statement for predicting.
        """
        return self._predict_anonymous_block

    def get_score_execute_statement(self):
        """
        Return the execute_statement for scoring.
        """
        return self._score_anonymous_block

    def get_parameters(self):
        """
        Parse sql lines containing the parameter definitions. In the sql code all the parameters
        are defined by four arrays, where the first one contains the parameter name, and one of the other
        three contains the value fitting to the parameter, while the other two are NULL. This format
        should be changed into a simple key-value based storage.

        Returns
        -------
            dict of list of tuples, where each tuple describes a parameter like (name, value, type)
        """
        result = {}
        if self._fit_param:
            result["fit"] = self._fit_param
        if self._predict_param:
            result["predict"] = self._predict_param
        if self._score_param:
            result["score"] = self._score_param
        return result

    def get_fit_parameters(self):
        """
        Get PAL fit parmeters.

        Returns
        -------
            List of tuples, where each tuple describes a parameter like (name, value, type)
        """
        return self._fit_param

    def get_predict_parameters(self):
        """
        Get PAL predict parmeters.

        Returns
        -------
            List of tuples, where each tuple describes a parameter like (name, value, type)
        """
        return self._predict_param

    def get_score_parameters(self):
        """
        Get PAL score parmeters.

        Returns
        -------
            List of tuples, where each tuple describes a parameter like (name, value, type)
        """
        return self._score_param

    def get_fit_output_table_names(self):
        """
        Get the generated result table names in fit function.

        Returns
        -------
            List of table names.
        """
        return self._fit_output_table_names

    def get_predict_output_table_names(self):
        """
        Get the generated result table names in predict function.

        Returns
        -------
            List of table names.
        """
        return self._predict_output_table_names

    def get_score_output_table_names(self):
        """
        Get the generated result table names in score function.

        Returns
        -------
            List of table names.
        """
        return self._score_output_table_names

    def _get_parameters(self):
        if "DECLARE group_id" in self.execute_statement:
            return _parse_params_with_group_id(_extract_params_definition_from_sql_with_group_id(self.execute_statement.split(";\n")))
        else:
            return _parse_params(_extract_params_definition_from_sql(self.execute_statement.split(";\n")))

    def get_pal_function(self):
        """
        Extract the specific function call of the PAL function from the sql code. Nevertheless it only detects
        the synonyms that have to be resolved afterwards

        Returns
        -------
        The procedure name synonym
        CALL "SYS_AFL.PAL_RANDOM_FORREST" (...) -> SYS_AFL.PAL_RANDOM_FORREST"
        """
        result = {}
        if self._fit_call:
            result["fit"] = self._fit_call
        if self._predict_call:
            result["predict"] = self._predict_call
        if self._score_call:
            result["score"] = self._score_call
        return result

    def _get_pal_function(self):
        if self.execute_statement:
            for line in self.execute_statement.split("\n"):
                calls = re.findall('CALL (.+)', line)
                if len(calls) > 0:
                    return calls[0]
        return None

    def _extract_output_table_names(self):
        sql = self.execute_statement.split(";\n")
        start_index, end_index = None, None
        for i, line in enumerate(sql):
            if re.match("CREATE LOCAL TEMPORARY COLUMN TABLE .+", line) and not start_index:
                start_index = i
            if re.match("END", line):
                end_index = i
                break
        if start_index is None:
            start_index = end_index
        res = []
        for line in sql[start_index:end_index]:
            res.append(re.findall(r'"(.*?)"', line)[0])
        return res

    def _call_pal_auto(self, conn_context, funcname, *args):
        self.runtime = None
        start_time = time.time()
        cast_dict = {}
        _args = list(args)
        if self._convert_bigint:
            for idx, _arg in enumerate(_args):
                if isinstance(_arg, DataFrame):
                    for col_name, col_type in _arg.get_table_structure().items():
                        if 'BIGINT' in col_type :
                            cast_dict[col_name] = 'DOUBLE'
                            logger.warning("%s has been cast from %s to DOUBLE", col_name, col_type)
                    _args[idx] = _arg.cast(cast_dict)
        self.execute_statement = call_pal_auto_with_hint(conn_context,
                                                        self.with_hint,
                                                        funcname,
                                                        *_args)

        self.runtime = time.time() - start_time
        try:
            call_string = self._get_pal_function()
            if call_string:
                if "INFERENCE" in call_string.upper() or "PREDICT" in call_string.upper() or "FORECAST" in call_string.upper() or "CLASSIFY" in call_string.upper() or "EXPLAIN" in call_string.upper():
                    self._predict_anonymous_block = self.execute_statement
                    self._predict_call = call_string
                    self._predict_param = self._get_parameters()
                    self._predict_output_table_names = self._extract_output_table_names()
                    self._predict_args = list(_args)
                elif "SCORE" in call_string.upper():
                    self._score_anonymous_block = self.execute_statement
                    self._score_call = call_string
                    self._score_param = self._get_parameters()
                    self._score_output_table_names = self._extract_output_table_names()
                    self._score_args = list(_args)
                else:
                    self._fit_anonymous_block = self.execute_statement
                    self._fit_call = call_string
                    self._fit_param = self._get_parameters()
                    self._fit_output_table_names = self._extract_output_table_names()
                    self._fit_args = list(_args)
        except Exception as err:
            logger.warning(err)
            pass

    def load_model(self, model):
        """
        Function to load fitted model.

        Parameters
        ----------
        model : DataFrame
            HANA DataFrame for fitted model.
        """
        self.model_ = model

    def _load_model_tables(self, schema_name, model_table_names, name, version, conn_context=None):
        """
        Function to load models.
        """
        if conn_context is None:
            conn_context = self.conn_context
        if isinstance(model_table_names, str):
            self.model_ = conn_context.table(model_table_names, schema=schema_name)
        elif isinstance(model_table_names, list):
            self.model_ = []
            for model_name in model_table_names:
                self.model_.append(conn_context.table(model_name, schema=schema_name))
        else:
            raise ValueError('Cannot load the model table. Unknwon values ({}, \
            {})'.format(schema_name, str(model_table_names)))

    def add_attribute(self, attr_key, attr_val):
        """
        Function to add attribute.
        """
        setattr(self, attr_key, attr_val)

def _parse_line(sql):
    return re.findall(":= (?:N')?([0-9A-Za-z_. \\[\\],{}-]+)'?", sql)[0]

def _extract_params_definition_from_sql(sql):
    start_index, end_index = None, None
    for i, line in enumerate(sql):
        if re.match("param_name\\[[1-9]+\\] := .+", line) and not start_index:
            start_index = i
        if re.match("params = UNNEST(.+)", line):
            end_index = i
            break
    if start_index is None:
        start_index = end_index
    return sql[start_index:end_index]

def _extract_params_definition_from_sql_with_group_id(sql):
    start_index, end_index = None, None
    for i, line in enumerate(sql):
        if re.match("group_id\\[[1-9]+\\] := .+", line) and not start_index:
            start_index = i
        if re.match("params = UNNEST(.+)", line):
            end_index = i
            break
    if start_index is None:
        start_index = end_index
    return sql[start_index:end_index]

def _parse_params_with_group_id(param_sql_raw):
    params = []
    param_names = []
    for i in range(0, len(param_sql_raw), 5):
        group_id = _parse_line(param_sql_raw[i])
        name = _parse_line(param_sql_raw[i + 1])
        param_i = _parse_line(param_sql_raw[i + 2])
        param_d = _parse_line(param_sql_raw[i + 3])
        param_s = _parse_line(param_sql_raw[i + 4])
        if param_i == 'NULL' and param_d == 'NULL':
            if name not in param_names:
                params.append((group_id, name, param_s, "string"))
                param_names.append(name)
            else:
                params[param_names.index(name)] = (group_id, name, params[param_names.index(name)][1] + ',' + param_s, "string")
        elif param_i == 'NULL' and param_s == 'NULL':
            params.append((group_id, name, float(param_d), "float"))
            param_names.append(name)
        elif param_d == 'NULL' and param_s == 'NULL':
            params.append((group_id, name, int(param_i), "integer"))
            param_names.append(name)
    return params

def _parse_params(param_sql_raw):
    params = []
    param_names = []
    for i in range(0, len(param_sql_raw), 4):
        name = _parse_line(param_sql_raw[i])
        param_i = _parse_line(param_sql_raw[i + 1])
        param_d = _parse_line(param_sql_raw[i + 2])
        param_s = _parse_line(param_sql_raw[i + 3])
        if param_i == 'NULL' and param_d == 'NULL':
            if name not in param_names:
                params.append((name, param_s, "string"))
                param_names.append(name)
            else:
                params[param_names.index(name)] = (name, params[param_names.index(name)][1] + ',' + param_s, "string")
        elif param_i == 'NULL' and param_s == 'NULL':
            params.append((name, float(param_d), "float"))
            param_names.append(name)
        elif param_d == 'NULL' and param_s == 'NULL':
            params.append((name, int(param_i), "integer"))
            param_names.append(name)
    return params

def attempt_version_comparison(minimum, actual):
    """
    Make our best guess at checking whether we have a high-enough version.

    This may not be a reliable comparison. The version number format has
    changed before, and it may change again. It is unclear what comparison,
    if any, would be reliable.

    Parameters
    ----------
    minimum : str
        (The first three components of) the version string for the
        minimum acceptable HANA version.
    actual : str
        The actual HANA version string.

    Returns
    -------
    bool
        True if (we think) the version is okay.
    """
    truncated_actual = actual.split()[0]
    min_as_ints = [int(x) for x in minimum.split('.')]
    actual_as_ints = [int(x) for x in truncated_actual.split('.')]
    return min_as_ints <= actual_as_ints

def require_pal_usable(conn):
    """
    Raises an error if no compatible PAL version is usable.

    To pass this check, HANA must be version 2 SPS 03,
    PAL must be installed, and the user must have one of the roles
    required to execute PAL procedures (AFL__SYS_AFL_AFLPAL_EXECUTE
    or AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION).

    A successful result is cached, to avoid redundant checks.

    Parameters
    ----------
    conn : ConnectionContext
        ConnectionContext on which PAL must be available.

    Raises
    ------
    hana_ml.ml_exceptions.PALUnusableError
        If the wrong HANA version is installed, PAL is uninstalled,
        or PAL execution permission is unavailable.
    """
    # pylint: disable=protected-access
    if not conn._pal_check_passed:
        with conn.connection.cursor() as cur:
            # Check HANA version. (According to SAP note 1898497, this
            # should match the PAL version.)
            cur.execute(_SELECT_HANA_VERSION)
            hana_version_string = cur.fetchone()[0]

            if not attempt_version_comparison(
                    minimum=MINIMUM_HANA_VERSION_PREFIX,
                    actual=hana_version_string):
                template = ('hana_ml version {} PAL support is not ' +
                            'compatible with this version of HANA. ' +
                            'HANA version must be at least {!r}, ' +
                            'but actual version string was {!r}.')
                msg = template.format(hana_ml.__version__,
                                      MINIMUM_HANA_VERSION_PREFIX,
                                      hana_version_string)
                raise hana_ml.ml_exceptions.PALUnusableError(msg)

            # Check PAL installation.
            cur.execute(_SELECT_PAL)
            if cur.fetchone() is None:
                raise hana_ml.ml_exceptions.PALUnusableError('PAL is not installed.')

            # Check required role.
            cur.execute(_SELECT_PAL_PRIVILEGE)
            if cur.fetchone() is None:
                msg = ('Missing needed role - PAL procedure execution ' +
                       'needs role AFL__SYS_AFL_AFLPAL_EXECUTE or ' +
                       'AFL__SYS_AFL_AFLPAL_EXECUTE_WITH_GRANT_OPTION')
                raise hana_ml.ml_exceptions.PALUnusableError(msg)
        conn._pal_check_passed = True

def call_pal(conn, funcname, *tablenames):
    """
    Call a PAL function.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection.
    funcname : str
        PAL procedure name.
    tablenames : list of str
        Table names to pass to PAL.
    """
    # This currently takes function names as "PAL_KMEANS".
    # Should that just be "KMEANS"?

    # callproc doesn't seem to handle table parameters.
    # It looks like we have to use execute.

    # In theory, this function should only be called with function and
    # table names that are safe without quoting.
    # We quote them anyway, for a bit of extra safety in case things
    # change or someone makes a typo in a call site.
    header = 'CALL _SYS_AFL.{}('.format(quotename(funcname))
    arglines_nosep = ['    {}'.format(quotename(tabname))
                      for tabname in tablenames]
    arglines_string = ',\n'.join(arglines_nosep)
    footer = ') WITH OVERVIEW'
    call_string = '{}\n{}\n{}'.format(header, arglines_string, footer)

    # SQLTRACE
    conn.sql_tracer.trace_object({
        'name':funcname,
        'schema': '_SYS_AFL',
        'type': 'pal'
    }, sub_cat='function')

    with conn.connection.cursor() as cur:
        execute_logged(cur, call_string, conn.sql_tracer, conn) # SQLTRACE added sql_tracer

def anon_block_safe(*dataframes):
    """
    Checks if these dataframes are compatible with call_pal_auto_with_hint.

    Parameters
    ----------
    df1, df2, ... : DataFrame
        DataFrames to be fed to PAL.

    Returns
    -------
    bool
        True if call_pal_auto_with_hintcan be used.
    """
    # pylint:disable=protected-access
    return all(df._ttab_handling in ('safe', 'ttab') for df in dataframes)

def call_pal_auto(conn,
                  funcname,
                  *args):
    """
    Uses an anonymous block to call a PAL function.

    DataFrames that are not known to be safe in anonymous blocks will be
    temporarily materialized.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection to use.
    funcname : str
        Name of the PAL function to execute. Should not include the "_SYS_AFL."
        part.
    arg1, arg2, ..., argN : DataFrame, ParameterTable, or str
        Arguments for the PAL function, in the same order the PAL function
        takes them.
        DataFrames represent input tables, a ParameterTable object represents
        the parameter table, and strings represent names for output
        tables. Output names with a leading "#" will produce local temporary
        tables.
    """
    args = list(args)
    return call_pal_auto_with_hint(conn, None, funcname, *args)

def call_pal_auto_with_hint(conn,
                            with_hint,
                            funcname,
                            *args):
    """
    Uses an anonymous block to call a PAL function.

    DataFrames that are not known to be safe in anonymous blocks will be
    temporarily materialized.

    Parameters
    ----------
    conn : ConnectionContext
        HANA connection to use.
    with_hint : str,
        If 'PARALLEL_BY_PARAMETER_PARTITIONS(p1)', it will use parallel with hint PARALLEL_BY_PARAMETER_PARTITIONS.
    funcname : str
        Name of the PAL function to execute. Should not include the "_SYS_AFL."
        part.
    arg1, arg2, ..., argN : DataFrame, ParameterTable, or str
        Arguments for the PAL function, in the same order the PAL function
        takes them.
        DataFrames represent input tables, a ParameterTable object represents
        the parameter table, and strings represent names for output
        tables. Output names with a leading "#" will produce local temporary
        tables.
    """
    adjusted_args = list(args)
    temporaries = []
    unknown_indices = []

    def materialize_at(i):
        "Materialize the i'th element of adjusted_args."
        tag = str(uuid.uuid4()).upper().replace('-', '_')
        name = '#{}_MATERIALIZED_INPUT_{}'.format(funcname, tag)
        adjusted_args[i] = adjusted_args[i].save(name)
        temporaries.append(name)

    def try_exec(cur, sql):
        """
        Try to execute the given sql. Returns True on success, False if
        execution fails due to an anonymous block trying to read a local
        temporary table. Other exceptions are propagated.
        """
        try:
            cur.execute(sql)
            return True
        except dbapi.Error as err:
            if not err.errortext.startswith(
                    'feature not supported: Cannot use local temporary table'):
                raise
            if "has invalid SQL type" in str(err):
                logger.error(" [HINT] The error may be due to the unsupported type such as BIGINT. Try to use enable_convert_bigint() to allow the bigint conversion.")
            return False
        except pyodbc.Error as err:
            if 'feature not supported: Cannot use local temporary table' in str(err.args[1]):
                raise
            if "has invalid SQL type" in str(err):
                logger.error(" [HINT] The error may be due to the unsupported type such as BIGINT. Try to use enable_convert_bigint() to allow the bigint conversion.")
            return False

    try:
        for i, argument in enumerate(args):
            if isinstance(argument, DataFrame):
                # pylint: disable=protected-access
                if argument._ttab_handling == 'unknown':
                    unknown_indices.append(i)
                elif argument._ttab_handling == 'unsafe':
                    materialize_at(i)

        # SQLTRACE added sql_tracer
        sql = sqlgen.call_pal_tabvar(funcname,
                                     conn.sql_tracer,
                                     with_hint,
                                     *adjusted_args)

        # SQLTRACE
        conn.sql_tracer.trace_object({
            'name':funcname,
            'schema': '_SYS_AFL',
            'type': 'pal'
        }, sub_cat='function')

        # Optimistic execution.
        with conn.connection.cursor() as cur:
            if try_exec(cur, sql):
                # Optimistic execution succeeded, meaning all arguments with
                # unknown ttab safety are safe.
                for i in unknown_indices:
                    adjusted_args[i].declare_lttab_usage(False)
                logged_without_execute(sql, conn.sql_tracer, conn)
                return sql

        # If we reach this point, optimistic execution failed.

        if len(unknown_indices) == 1:
            # Only one argument of unknown ttab safety, so that one needs
            # materialization.
            adjusted_args[unknown_indices[0]].declare_lttab_usage(True)
            materialize_at(unknown_indices[0])
        else:
            # Multiple arguments of unknown safety. Test which ones are safe.
            for i in unknown_indices:
                with conn.connection.cursor() as cur:
                    ttab_used = not try_exec(cur, sqlgen.safety_test(adjusted_args[i]))
                adjusted_args[i].declare_lttab_usage(ttab_used)
                if ttab_used:
                    materialize_at(i)

        # SQLTRACE added sql_tracer
        sql = sqlgen.call_pal_tabvar(funcname,
                                     conn.sql_tracer,
                                     with_hint,
                                     *adjusted_args)
        with conn.connection.cursor() as cur:
            execute_logged(cur, sql, conn.sql_tracer, conn) # SQLTRACE added sql_tracer
        return sql
    finally:
        try_drop(conn, temporaries)
