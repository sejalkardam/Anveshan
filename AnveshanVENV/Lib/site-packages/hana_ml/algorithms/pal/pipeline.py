"""
This module supports to run PAL functions in a pipeline manner.
"""

#pylint: disable=invalid-name
#pylint: disable=eval-used
#pylint: disable=unused-variable
#pylint: disable=line-too-long
#pylint: disable=too-many-locals
#pylint: disable=too-many-branches
#pylint: disable=consider-using-f-string
import re
from hana_ml.visualizers.digraph import Digraph

class Pipeline(object): #pylint: disable=useless-object-inheritance
    """
    Pipeline construction to run transformers and estimators sequentially.

    Parameters
    ----------

    step : list
        List of (name, transform) tuples that are chained. The last object should be an estimator.
    """
    def __init__(self, steps):
        self.steps = steps
        self.nodes = []

    def fit_transform(self, data, fit_params=None):
        """
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------

        data : DataFrame
            SAP HANA DataFrame to be transformed in the pipeline.
        fit_params : dict
            The parameters corresponding to the transformers/estimator name
            where each parameter name is prefixed such that parameter p for step s has key s__p.

        Returns
        -------

        DataFrame
            Transformed SAP HANA DataFrame.

        Examples
        --------

        >>> my_pipeline = Pipeline([
                ('pca', PCA(scaling=True, scores=True)),
                ('imputer', Imputer(strategy='mean'))
                ])
        >>> fit_params = {'pca__key': 'ID', 'pca__label': 'CLASS'}
        >>> my_pipeline.fit_transform(data=train_data, fit_params=fit_params)

        """
        data_ = data
        count = 0
        if fit_params is None:
            fit_params = {}
        for step in self.steps:
            fit_param_str = ''
            m_fit_params = {}
            for param_key, param_val in fit_params.items():
                if "__" not in param_key:
                    raise ValueError("The parameter name format incorrect. The parameter name is prefixed such that parameter p for step s has key s__p.")
                step_marker, param_name = param_key.split("__")
                if step[0] in step_marker:
                    m_fit_params[param_name] = param_val
                    fit_param_str = fit_param_str + ",\n" + param_name + "="
                    if isinstance(param_val, str):
                        fit_param_str = fit_param_str + "'{}'".format(param_val)
                    else:
                        fit_param_str = fit_param_str + str(param_val)
            data_ = step[1].fit_transform(data_, **m_fit_params)
            self.nodes.append((step[0],
                               "{}.fit_transform(data={}{})".format(_get_obj(step[1]),
                                                                    repr(data_),
                                                                    fit_param_str),
                               [str(count)],
                               [str(count + 1)]))
            count = count + 1
        return data_

    def fit(self, data, fit_params=None):
        """
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.

        Parameters
        ----------
        data : DataFrame
            SAP HANA DataFrame to be transformed in the pipeline.
        fit_params : dict, optional
            Parameters corresponding to the transformers/estimator name
            where each parameter name is prefixed such that parameter p for step s has key s__p.

        Returns
        -------
        DataFrame
            Transformed SAP HANA DataFrame.

        Examples
        --------

        >>> my_pipeline = Pipeline([
            ('pca', PCA(scaling=True, scores=True)),
            ('imputer', Imputer(strategy='mean')),
            ('hgbt', HybridGradientBoostingClassifier(
            n_estimators=4, split_threshold=0, learning_rate=0.5, fold_num=5,
            max_depth=6, cross_validation_range=cv_range))
            ])
        >>> fit_params = {'pca__key': 'ID',
                          'pca__label': 'CLASS',
                          'hgbt__key': 'ID',
                          'hgbt__label': 'CLASS',
                          'hgbt__categorical_variable': 'CLASS'}
        >>> hgbt_model = my_pipeline.fit(data=train_data, fit_params=fit_params)
        """
        data_ = data
        count = 0
        if fit_params is None:
            fit_params = {}
        obj = None
        for step in self.steps:
            fit_param_str = ''
            m_fit_params = {}
            for param_key, param_val in fit_params.items():
                if "__" not in param_key:
                    raise ValueError("The parameter name format incorrect. The parameter name is prefixed such that parameter p for step s has key s__p.")
                step_marker, param_name = param_key.split("__")
                if step[0] in step_marker:
                    m_fit_params[param_name] = param_val
                    fit_param_str = fit_param_str + ",\n" + param_name + "="
                    if isinstance(param_val, str):
                        fit_param_str = fit_param_str + "'{}'".format(param_val)
                    else:
                        fit_param_str = fit_param_str + str(param_val)
            obj = step[1]
            if count < len(self.steps) - 1:
                data_ = obj.fit_transform(data_, **m_fit_params)
                self.nodes.append((step[0],
                                   "{}.fit_transform(data={}{})".format(_get_obj(obj),
                                                                        repr(data_),
                                                                        fit_param_str),
                                   [str(count)],
                                   [str(count + 1)]))
            else:
                obj.fit(data_, **m_fit_params)
                self.nodes.append((step[0],
                                   "{}.fit(data={}{})".format(_get_obj(obj),
                                                              repr(data_),
                                                              fit_param_str),
                                   [str(count)],
                                   [str(count + 1)]))
            count = count + 1
        return obj

    def fit_predict(self, data, apply_data=None, fit_params=None, predict_params=None):
        """
        Fit all the transforms one after the other and transform the
        data, then fit_predict the transformed data using the last estimator.

        Parameters
        ----------
        data : DataFrame
            SAP HANA DataFrame to be transformed in the pipeline.
        apply_data : DataFrame
            SAP HANA DataFrame to be predicted in the pipeline.
        fit_params : dict, optional
            Parameters corresponding to the transformers/estimator name
            where each parameter name is prefixed such that parameter p for step s has key s__p.
        predict_params : dict, optional
            Parameters corresponding to the predictor name
            where each parameter name is prefixed such that parameter p for step s has key s__p.

        Returns
        -------
        DataFrame
            Transformed SAP HANA DataFrame.

        Examples
        --------

        >>> my_pipeline = Pipeline([
            ('pca', PCA(scaling=True, scores=True)),
            ('imputer', Imputer(strategy='mean')),
            ('hgbt', HybridGradientBoostingClassifier(
            n_estimators=4, split_threshold=0, learning_rate=0.5, fold_num=5,
            max_depth=6, cross_validation_range=cv_range))
            ])
        >>> fit_params = {'pca__key': 'ID',
                          'pca__label': 'CLASS',
                          'hgbt__key': 'ID',
                          'hgbt__label': 'CLASS',
                          'hgbt__categorical_variable': 'CLASS'}
        >>> hgbt_model = my_pipeline.fit_predict(data=train_data, apply_data=test_data, fit_params=fit_params)
        """
        data_ = data
        count = 0
        if fit_params is None:
            fit_params = {}
        if predict_params is None:
            predict_params = {}
        for step in self.steps:
            fit_param_str = ''
            m_fit_params = {}
            m_predict_params = {}
            for param_key, param_val in fit_params.items():
                if "__" not in param_key:
                    raise ValueError("The parameter name format incorrect. The parameter name is prefixed such that parameter p for step s has key s__p.")
                step_marker, param_name = param_key.split("__")
                if step[0] in step_marker:
                    m_fit_params[param_name] = param_val
                    fit_param_str = fit_param_str + ",\n" + param_name + "="
                    if isinstance(param_val, str):
                        fit_param_str = fit_param_str + "'{}'".format(param_val)
                    else:
                        fit_param_str = fit_param_str + str(param_val)
            predit_param_str = ''
            for param_key, param_val in predict_params.items():
                if "__" not in param_key:
                    raise ValueError("The parameter name format incorrect. The parameter name is prefixed such that parameter p for step s has key s__p.")
                step_marker, param_name = param_key.split("__")
                if step[0] in step_marker:
                    m_predict_params[param_name] = param_val
                    predit_param_str = predit_param_str + ",\n" + param_name + "="
                    if isinstance(param_val, str):
                        predit_param_str = predit_param_str + "'{}'".format(param_val)
                    else:
                        predit_param_str = predit_param_str + str(param_val)
            if count < len(self.steps) - 1:
                data_ = step[1].fit_transform(data_, **m_fit_params)
                self.nodes.append((step[0],
                                   "{}\n.fit_transform(data={}{})".format(_get_obj(step[1]),
                                                                          repr(data_),
                                                                          fit_param_str),
                                   [str(count)],
                                   [str(count + 1)]))
            else:
                if apply_data:
                    data_ = step[1].fit(data_, **m_fit_params).predict(apply_data, **m_predict_params)
                else:
                    data_ = step[1].fit(data_, **m_fit_params).predict(**m_predict_params)
                if apply_data:
                    apply_param_str = repr(apply_data) + ", " + predit_param_str[2:]
                else:
                    apply_param_str = predit_param_str[2:]
                self.nodes.append((step[0],
                                   "{}\n.fit(data={}{})\n.predict({})".format(_get_obj(step[1]),
                                                                              repr(data_),
                                                                              fit_param_str,
                                                                              apply_param_str),
                                   [str(count)],
                                   [str(count + 1)]))
            count = count + 1
        return data_

    def plot(self, name="my_pipeline", iframe_height=450):
        """
        Plot pipeline.
        """
        digraph = Digraph(name)
        node = []
        for elem in self.nodes:
            node.append(digraph.add_python_node(elem[0],
                                                elem[1],
                                                in_ports=elem[2],
                                                out_ports=elem[3]))
        for node_x in range(0, len(node) - 1):
            digraph.add_edge(node[node_x].out_ports[0],
                             node[node_x + 1].in_ports[0])
        digraph.build()
        digraph.generate_notebook_iframe(iframe_height)

def _get_obj(obj):
    tmp_mem = []
    for key, val in obj.hanaml_parameters.items():
        if val is not None:
            tmp_mem.append("{}={}".format(key, val))
    return "{}({})".format(re.findall('\'([^$]*)\'', str(obj.__class__))[0], ",\n".join(tmp_mem))
