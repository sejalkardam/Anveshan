"""
This module represents a visualizer for tree model.
The following class is available:
    * :class:`TreeModelDebriefing`
"""

# pylint: disable=import-error
# pylint: disable=line-too-long
# pylint: disable=superfluous-parens
# pylint: disable=missing-class-docstring
# pylint: disable=too-many-locals
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# pylint: disable=missing-docstring
# pylint: disable=superfluous-parens
# pylint: disable=consider-using-f-string
import logging
import uuid
import html
import pydotplus
try:
    import pyodbc
except ImportError as error:
    pass
from hdbcli import dbapi
from hana_ml import dataframe
from hana_ml.ml_base import try_drop
from hana_ml.algorithms.pal.pal_base import (
    ParameterTable,
    call_pal_auto_with_hint
)
from hana_ml.visualizers.shap import ShapleyExplainer
from hana_ml.visualizers.digraph import DigraphConfig, BaseDigraph, Digraph, MultiDigraph
from hana_ml.visualizers.ui_components import JSONViewer, XMLViewer

logger = logging.getLogger(__name__)


class ModelDebriefingUtils(object):
    # Utility class for all model visualizations.
    def __init__(self):
        pass

    @staticmethod
    def is_pmml_format(data):
        """
        Determine whether the data format is PMML.

        Parameters
        ----------

        data : String

            Tree model data.

        Returns
        -------

        True or False

        """
        return data.endswith('</PMML>')

    @staticmethod
    def is_digraph_format(data):
        """
        Determine whether the data format is DOT.

        Parameters
        ----------

        data : String

            Tree model data.

        Returns
        -------

        True or False

        """
        return data.startswith('digraph')

    @staticmethod
    def check_dataframe_type(model):
        """
        Determine whether the type of model object is hana_ml.dataframe.DataFrame.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        Throw exception

        """
        if isinstance(model, dataframe.DataFrame) is False:
            raise TypeError("The type of parameter 'model' must be hana_ml.dataframe.DataFrame!")

    @staticmethod
    def warning(msg):
        """
        Print message with red color.

        Parameters
        ----------

        msg : String

            Message.
        """
        print('\033[31m{}'.format(msg))


class TreeModelDebriefing(object):
    r"""
    Visualize tree model.

    Examples
    --------

    Visualize Tree Model in JSON format:

    >>> TreeModelDebriefing.tree_debrief(rdt.model_)

    .. image:: json_model.png

    Visualize Tree Model in DOT format:

    >>> TreeModelDebriefing.tree_parse(rdt.model_)
    >>> TreeModelDebriefing.tree_debrief_with_dot(rdt.model_)

    .. image:: dot_model.png

    Visualize Tree Model in XML format the model is stored in the dataframe rdt.model\_:

    >>> treeModelDebriefing.tree_debrief(rdt.model_)

    .. image:: xml_model.png


    """

    def __init__(self):
        pass

    __PARSE_MODEL_FUC_NAME = 'PAL_VISUALIZE_MODEL'

    __TREE_INDEX_FIRST_VALUE = '0'
    __TREE_DICT_NAME = '_tree_dict'
    __TREE_DOT_DICT_NAME = '_tree_dot_dict'
    __TREE_UI_Viewer = '_tree_ui_viewer'
    __TREE_UI_Digraph = '_tree_ui_digraph'

    @staticmethod
    def __add_tree_dict(model):
        if model.__dict__.get(TreeModelDebriefing.__TREE_DICT_NAME) is None:
            model.__dict__[TreeModelDebriefing.__TREE_DICT_NAME] = TreeModelDebriefing.__parse_tree_model(model)

    @staticmethod
    def __check_tree_dot_dict(model):
        if model.__dict__.get(TreeModelDebriefing.__TREE_DOT_DICT_NAME) is None:
            raise AttributeError('You must parse the model firstly!')

    @staticmethod
    def __parse_tree_model(model):
        tree_dict = {}
        if len(model.columns) == 3:
            # multiple trees
            # |ROW_INDEX|TREE_INDEX|MODEL_CONTENT|
            for tree_index, single_tree_list in \
                    model.sort(model.columns[0]).collect().groupby(model.columns[1])[model.columns[2]]:
                tree_dict[str(tree_index)] = "".join(single_tree_list)
        else:
            # single tree
            # |ROW_INDEX|MODEL_CONTENT|
            tree_dict[TreeModelDebriefing.__TREE_INDEX_FIRST_VALUE] = "".join(model.sort(model.columns[0]).collect()[model.columns[1]])

        return tree_dict

    @staticmethod
    def tree_debrief(model):
        """
        Visualize tree model by data in JSON or XML format.

        Parameters
        ----------

        model : DataFrame

            Tree model.

        Returns
        -------

        HTML Page

            This HTML page can be rendered by browser.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        TreeModelDebriefing.__add_tree_dict(model)
        tree_dict = model.__dict__[TreeModelDebriefing.__TREE_DICT_NAME]

        if len(model.columns) == 3:
            # multiple trees
            tree_keys = list(tree_dict.keys())
            if ModelDebriefingUtils.is_pmml_format(tree_dict[tree_keys[0]]):
                # xml format
                xml_viewer = XMLViewer(tree_dict)
                xml_viewer.generate_notebook_iframe()
                model.__dict__[TreeModelDebriefing.__TREE_UI_Viewer] = xml_viewer
            else:
                # json format
                json_viewer = JSONViewer(tree_dict)
                json_viewer.generate_notebook_iframe()
                model.__dict__[TreeModelDebriefing.__TREE_UI_Viewer] = json_viewer
        else:
            # single tree
            if ModelDebriefingUtils.is_pmml_format(tree_dict[TreeModelDebriefing.__TREE_INDEX_FIRST_VALUE]):
                # xml format
                xml_viewer = XMLViewer(tree_dict)
                xml_viewer.generate_notebook_iframe()
                model.__dict__[TreeModelDebriefing.__TREE_UI_Viewer] = xml_viewer
            else:
                # json format
                json_viewer = JSONViewer(model.__dict__[TreeModelDebriefing.__TREE_DICT_NAME])
                json_viewer.generate_notebook_iframe()
                model.__dict__[TreeModelDebriefing.__TREE_UI_Viewer] = json_viewer

    @staticmethod
    def tree_export(model, filename):
        """
        Save the tree model as a html file.

        Parameters
        ----------
        model : DataFrame
            Tree model.
        filename : str
            Html file name.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        viewer = model.__dict__.get(TreeModelDebriefing.__TREE_UI_Viewer)
        if viewer is None:
            raise Exception("You must call 'tree_debrief' method firstly!")
        else:
            viewer.generate_html(filename)

    @staticmethod
    def tree_parse(model):
        """
        Transform tree model content using DOT language.

        Parameters
        ----------

        model : DataFrame

            Tree model.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        dot_tbl_name = '#PAL_DT_DOT_TBL_{}'.format(str(uuid.uuid1()).replace('-', '_').upper())
        tables = [dot_tbl_name]
        param_rows = []
        try:
            call_pal_auto_with_hint(model.connection_context,
                                    None,
                                    TreeModelDebriefing.__PARSE_MODEL_FUC_NAME,
                                    model,
                                    ParameterTable().with_data(param_rows),
                                    *tables)

            model.__dict__[TreeModelDebriefing.__TREE_DOT_DICT_NAME] = TreeModelDebriefing.__parse_tree_model(model.connection_context.table(dot_tbl_name))
        except dbapi.Error as db_err:
            logger.exception(str(db_err))
            try_drop(model.connection_context, dot_tbl_name)
            raise
        except pyodbc.Error as db_err:
            logger.exception(str(db_err.args[1]))
            try_drop(model.connection_context, dot_tbl_name)
            raise

    @staticmethod
    def tree_debrief_with_dot(model, iframe_height: int = 800, digraph_config: DigraphConfig = None):
        """
        Visualize tree model by data in DOT format.

        Parameters
        ----------
        model : DataFrame
            Tree model.
        iframe_height : int, optional
            Frame height.

            Defaults to 800.
        digraph_config : DigraphConfig, optional
            Configuration instance of digraph.

        Returns
        -------
        HTML Page
            This HTML page can be rendered by browser.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        TreeModelDebriefing.__check_tree_dot_dict(model)

        if digraph_config is None:
            digraph_config = DigraphConfig()
            digraph_config.set_digraph_layout('vertical')

        if len(model.columns) == 3:
            # multiple trees
            multi_digraph = MultiDigraph('Tree Model')
            tree_list = list(model.__dict__[TreeModelDebriefing.__TREE_DOT_DICT_NAME].keys())
            for tree_index in tree_list:
                dot_data = model.__dict__[TreeModelDebriefing.__TREE_DOT_DICT_NAME][str(tree_index)]
                child_digraph: MultiDigraph.ChildDigraph = multi_digraph.add_child_digraph('Tree_'+str(tree_index))
                TreeModelDebriefing.__add_model_node_with_dot(child_digraph, dot_data)
            multi_digraph.build(digraph_config)
            multi_digraph.generate_notebook_iframe(iframe_height)
            model.__dict__[TreeModelDebriefing.__TREE_UI_Digraph] = multi_digraph
        else:
            # single tree
            dot_data = model.__dict__[TreeModelDebriefing.__TREE_DOT_DICT_NAME][TreeModelDebriefing.__TREE_INDEX_FIRST_VALUE]
            digraph = Digraph('Tree Model')
            TreeModelDebriefing.__add_model_node_with_dot(digraph, dot_data)
            digraph.build(digraph_config)
            digraph.generate_notebook_iframe(iframe_height)
            model.__dict__[TreeModelDebriefing.__TREE_UI_Digraph] = digraph

    @staticmethod
    def __add_model_node_with_dot(digraph: BaseDigraph, dot_data: str) -> None:
        dot_graph = pydotplus.graph_from_dot_data(dot_data.encode('utf8'))
        node_id_2_node_dict = {}
        node_id_2_in_port_dict = {}
        node_id_2_out_port_dict = {}
        edges = []

        for edge in dot_graph.get_edges():
            node_id_2_out_port_dict[edge.get_source()] = 'out'
            node_id_2_in_port_dict[edge.get_destination()] = 'in'
            edges.append((edge.get_source(), edge.get_destination()))

        nodes = dot_graph.get_nodes()
        for node in nodes:
            node_label = node.get_label()
            if node_label:
                node_name = html.unescape(node_label[1:-1].replace('<br/>', '\n'))
                in_port = []
                if node_id_2_in_port_dict.get(node.get_name()):
                    in_port = ['in']
                out_port = []
                if node_id_2_out_port_dict.get(node.get_name()):
                    out_port = ['out']
                added_node = digraph.add_model_node(node_name, '', in_port, out_port)
                node_id_2_node_dict[node.get_name()] = added_node

        # node has only: [1 in and 1 out] port
        for item in edges:
            node1 = node_id_2_node_dict.get(item[0])
            node2 = node_id_2_node_dict.get(item[1])
            digraph.add_edge(node1.out_ports[0], node2.in_ports[0])

    @staticmethod
    def tree_export_with_dot(model, filename):
        """
        Save the tree model as a html file.

        Parameters
        ----------
        model : DataFrame
            Tree model.
        filename : str
            Html file name.
        """
        ModelDebriefingUtils.check_dataframe_type(model)
        viewer = model.__dict__.get(TreeModelDebriefing.__TREE_UI_Digraph)
        if viewer is None:
            raise Exception("You must call 'tree_debrief_with_dot' method firstly!")
        else:
            viewer.generate_html(filename)

    @staticmethod
    def shapley_explainer(predict_result: dataframe.DataFrame, predict_data: dataframe.DataFrame, key=None, label=None, predict_reason_column='REASON_CODE'):
        """
        Create Shapley explainer to explain the output of machine learning model. \n
        It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.

        Parameters
        ----------
        predict_result : DataFrame
            Predicted result.

        predict_data : DataFrame
            Predicted dataset.

        key : str
            Name of the ID column.

        label : str
            Name of the dependent variable.

        predict_reason_column : str
            Predicted result, structured as follows:
                -  column : REASON CODE, valid only for tree-based functionalities.

        Returns
        -------
        :class:`~hana_ml.visualizers.shap.ShapleyExplainer`
            Shapley explainer.
        """
        return ShapleyExplainer(predict_result, predict_data, key, label, predict_reason_column)
