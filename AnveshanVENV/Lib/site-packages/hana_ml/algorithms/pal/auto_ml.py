"""
This module contains auto machine learning API.

"""
#pylint: disable=too-many-branches
#pylint: disable=too-many-statements
#pylint: disable=too-many-locals
#pylint: disable=line-too-long
#pylint: disable=too-few-public-methods
#pylint: disable=consider-using-f-string
from hana_ml.algorithms.pal.pal_base import PALBase
from hana_ml.algorithms.pal.preprocessing import FeatureNormalizer
from hana_ml.algorithms.pal.preprocessing import KBinsDiscretizer
from hana_ml.algorithms.pal.preprocessing import Imputer
from hana_ml.algorithms.pal.preprocessing import Discretize
from hana_ml.algorithms.pal.preprocessing import MDS
from hana_ml.algorithms.pal.preprocessing import SMOTE
from hana_ml.algorithms.pal.preprocessing import SMOTETomek
from hana_ml.algorithms.pal.preprocessing import TomekLinks
from hana_ml.algorithms.pal.preprocessing import Sampling
from hana_ml.algorithms.pal.preprocessing import FeatureSelection
from hana_ml.algorithms.pal.decomposition import PCA

class Preprocessing(PALBase):
    """
    Preprocessing class. Similar to the function preprocessing.

    Parameters
    ----------
    name : str
        The preprocessing algorithm name.

        - OneHotEncoder
        - FeatureNormalizer
        - KBinsDiscretizer
        - Imputer
        - Discretize
        - MDS
        - SMOTE
        - SMOTETomek
        - TomekLinks
        - Sampling
    **kwargs: dict
        A dict of the keyword args passed to the object.
        Please refer to the documentation of the specific preprocessing for parameter information.
    """
    def __init__(self, name, **kwargs):
        super(Preprocessing, self).__init__()
        self.name = name
        self.kwargs = dict(**kwargs)

    def fit_transform(self, data, key=None, features=None, **kwargs):
        """
        Execute the preprocessing algorithm and return the transformed dataset.

        Parameters
        ----------
        data : DataFrame
            Input data.
        key : str, optional
            Name of the ID column.

            Defaults to the index column of ``data`` (i.e. data.index) if it is set.
        features : list, optional
            The columns to be preprocessed.
        **kwargs: dict
            A dict of the keyword args passed to the fit_transform function.
            Please refer to the documentation of the specific preprocessing for parameter information.
        """
        args = dict(**kwargs)
        if data.index is not None:
            key = data.index
        key_is_none = False
        if key is None:
            key_is_none = True
        if features is None:
            features = data.columns
            if self.name == 'OneHotEncoder':
                features = []
                for col, val in data.get_table_structure().items():
                    if 'VARCHAR' in val.upper():
                        features.append(col)
            if key is not None:
                if key in features:
                    features.remove(key)
        if isinstance(features, str):
            features = [features]
        if self.name != "OneHotEncoder":
            if key is None:
                key = "ID"
                data = data.add_id(key)
            data = data.select([key] + features)
        other = data.deselect(features)
        if self.name == 'FeatureNormalizer':
            if 'method' not in self.kwargs.keys():
                self.kwargs['method'] = "min-max"
                self.kwargs['new_max'] = 1.0
                self.kwargs['new_min'] = 0.0
            transformer = FeatureNormalizer(**self.kwargs)
            result = transformer.fit_transform(data, key, **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'KBinsDiscretizer':
            if 'strategy' not in self.kwargs.keys():
                self.kwargs['strategy'] = "uniform_size"
                self.kwargs['smoothing'] = "means"
            transformer = KBinsDiscretizer(**self.kwargs)
            result = transformer.fit_transform(data, key, **args).deselect(["BIN_INDEX"])
            self.execute_statement = transformer.execute_statement
        elif self.name == 'Imputer':
            transformer = Imputer(**self.kwargs)
            result = transformer.fit_transform(data,
                                               key,
                                               **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'Discretize':
            if 'strategy' not in self.kwargs.keys():
                self.kwargs['strategy'] = "uniform_size"
                self.kwargs['smoothing'] = "bin_means"
            transformer = Discretize(**self.kwargs)
            if 'binning_variable' not in args.keys():
                args['binning_variable'] = features
            result = transformer.fit_transform(data,
                                               **args)[0]
            self.execute_statement = transformer.execute_statement
        elif self.name == 'MDS':
            if 'matrix_type' not in self.kwargs.keys():
                self.kwargs['matrix_type'] = "observation_feature"
            transformer = MDS(**self.kwargs)
            result = transformer.fit_transform(data, key, **args)
            result = result[0].pivot_table(values='VALUE', index='ID', columns='DIMENSION', aggfunc='avg')
            columns = result.columns
            rename_cols = {}
            for col in columns:
                if col != "ID":
                    rename_cols[col] = "X_" + str(col)
            result = result.rename_columns(rename_cols)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'SMOTE':
            transformer = SMOTE(**self.kwargs)
            result = transformer.fit_transform(data, **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'SMOTETomek':
            transformer = SMOTETomek(**self.kwargs)
            result = transformer.fit_transform(data,
                                               **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'TomekLinks':
            transformer = TomekLinks(**self.kwargs)
            result = transformer.fit_transform(data,
                                               key=key,
                                               **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'Sampling':
            transformer = Sampling(**self.kwargs)
            result = transformer.fit_transform(data, **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'PCA':
            transformer = PCA(**self.kwargs)
            result = transformer.fit_transform(data,
                                               key=key,
                                               **args)
            self.execute_statement = transformer.execute_statement
        elif self.name == 'OneHotEncoder':
            others = list(set(data.columns) - set(features))
            query = "SELECT {}".format(", ".join(others))
            if others:
                query = query + ", "
            for feature in features:
                categoricals = data.distinct(feature).collect()[feature].to_list()
                for cat in categoricals:
                    query = query + "CASE WHEN \"{0}\" = '{1}' THEN 1 ELSE 0 END \"{1}_{0}_OneHot\", ".format(feature, cat)
            query = query[:-2] + " FROM ({})".format(data.select_statement)
            self.execute_statement = query
            return data.connection_context.sql(query)
        elif self.name == 'FeatureSelection':
            transformer = FeatureSelection(**self.kwargs)
            result = transformer.fit_transform(data,
                                               key,
                                               **args)
            self.execute_statement = transformer.execute_statement
        else:
            pass
        if features is not None:
            if key is not None:
                result = other.set_index(key).join(result.set_index(key))
        if key_is_none:
            result = result.deselect(key)
        return result
