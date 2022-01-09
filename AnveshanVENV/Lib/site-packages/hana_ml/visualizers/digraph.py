"""
This module represents the whole digraph framework.
The whole digraph framework consists of Python API and page assets(HTML, CSS, JS, Font, Icon, etc.).
The application scenarios of the current digraph framework are AutoML Pipeline and Model Debriefing.

The following classes are available:
    * :class:`Node`
    * :class:`InPort`
    * :class:`OutPort`
    * :class:`Edge`
    * :class:`DigraphConfig`
    * :class:`Digraph`
    * :class:`MultiDigraph`
    * :class:`MultiDigraph.ChildDigraph`
"""

# pylint: disable=too-many-arguments
# pylint: disable=too-few-public-methods
# pylint: disable=line-too-long
# pylint: disable=too-many-locals
# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-variable
# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=trailing-newlines
# pylint: disable=useless-super-delegation
# pylint: disable=undefined-variable
# pylint: disable=consider-using-f-string
import time
from threading import Lock
from typing import List
from hana_ml.visualizers.model_report import TemplateUtil
from hana_ml.visualizers.ui_components import Fullscreen, HTMLFrameUtils, HTMLUtils, build_html_exception_msg


class Node(object):
    """
    The Node class of digraph framework is an entity class.

    Parameters
    ----------
    node_id : int [Automatic generation]
        Unique identification of node.
    node_name : str
        The node name.
    node_icon_id : int [Automatic generation]
        Unique identification of node icon.
    node_content : str
        The node content.
    node_in_ports : list
        List of input port names.
    node_out_ports : list
        List of output port names.
    """
    def __init__(self, node_id: int, node_name: str, node_icon_id: int, node_content: str, node_in_ports: list, node_out_ports: list):
        self.id: int = node_id
        self.name: str = node_name
        self.icon_id: int = node_icon_id
        self.content: str = node_content
        self.in_ports: List[InPort] = []
        self.out_ports: List[OutPort] = []

        sequence = 0
        for in_port_name in node_in_ports:
            sequence = sequence + 1
            self.in_ports.append(InPort(self, '{}_in_{}'.format(node_id, sequence), in_port_name, sequence))

        sequence = 0
        for out_port_name in node_out_ports:
            sequence = sequence + 1
            self.out_ports.append(OutPort(self, '{}_out_{}'.format(node_id, sequence), out_port_name, sequence))

        in_ports = []
        out_ports = []
        for in_port in self.in_ports:
            in_ports.append(in_port.json_data)
        for out_port in self.out_ports:
            out_ports.append(out_port.json_data)

        self.json_data: dict = {
            'id': '{}'.format(self.id),
            'name': self.name,
            'script': self.content,
            'iconId': self.icon_id,
            'inPorts': in_ports,
            'outPorts': out_ports
        }


class InPort(object):
    """
    The InPort class of digraph framework is an entity class. \n
    A port is a fixed connection point on a node.

    Parameters
    ----------
    node : Node
        Which node is the input port fixed on.
    port_id : str [Automatic generation]
        Unique identification of input port.
    port_name : str
        The input port name.
    port_sequence : int [Automatic generation]
        The position of input port among all input ports.
    """
    def __init__(self, node: Node, port_id: str, port_name: str, port_sequence: int):
        self.type: str = 'in'

        self.node: Node = node
        self.id: str = port_id
        self.name: str = port_name
        self.sequence: int = port_sequence

        self.json_data: dict = {
            'id': self.id,
            'name': self.name,
            'sequence': self.sequence
        }


class OutPort(object):
    """
    The OutPort class of digraph framework is an entity class. \n
    A port is a fixed connection point on a node.

    Parameters
    ----------
    node : Node
        Which node is the output port fixed on.
    port_id : str [Automatic generation]
        Unique identification of output port.
    port_name : str
        The output port name.
    port_sequence : int [Automatic generation]
        The position of output port among all output ports.
    """
    def __init__(self, node: Node, port_id: str, port_name: str, port_sequence: int):
        self.type: str = 'out'

        self.node: Node = node
        self.id: str = port_id
        self.name: str = port_name
        self.sequence: int = port_sequence

        self.json_data: dict = {
            'id': self.id,
            'name': self.name,
            'sequence': self.sequence
        }


class Edge(object):
    """
    The Edge class of digraph framework is an entity class. \n
    The output port of a node is connected with the input port of another node to make an edge.

    Parameters
    ----------
    source_port : OutPort
        Start connection point of edge.
    target_port : InPort
        End connection point of edge.
    """
    def __init__(self, source_port: OutPort, target_port: InPort):
        if source_port is None or target_port is None:
            raise Exception('The passed parameter is invalid!')

        self.source_port: OutPort = source_port
        self.target_port: InPort = target_port
        self.json_data: dict = {
            'source': '{}'.format(self.source_port.node.id),
            'target': '{}'.format(self.target_port.node.id),
            'outputPortId': self.source_port.id,
            'inputPortId': self.target_port.id
        }


class DigraphConfig(object):
    """
    Configuration class of digraph.
    """
    def __init__(self):
        self.options: dict = {
            'makeTextCenter': 'false',
            'digraphLayout': 'horizontal',
            'nodeSep': 80,
            'rankSep': 80
        }

    def set_text_layout(self, make_text_center: bool = False):
        """
        Set node's text layout.

        Parameters
        ----------
        make_text_center : bool, optional
            Should the node's text be centered.

            Defaults to False.
        """
        if make_text_center:
            self.options['makeTextCenter'] = 'true'
        else:
            self.options['makeTextCenter'] = 'false'

    def set_digraph_layout(self, digraph_layout: str = 'horizontal'):
        """
        Set the layout of a digraph.

        Parameters
        ----------
        digraph_layout : str, optional
            The layout of a digraph can only be horizontal or vertical.

            Defaults to horizontal layout.
        """
        if digraph_layout in ('horizontal', 'vertical'):
            self.options['digraphLayout'] = digraph_layout
        else:
            raise ValueError('The layout of a digraph can only be horizontal or vertical!')

    def set_node_sep(self, node_sep: int = 80):
        """
        Set distance between nodes.\n
        Under horizontal layout, this parameter represents horizontal distance between nodes.\n
        Under vertical layout, this parameter represents vertical distance between nodes.

        Parameters
        ----------
        node_sep : int, optional
            The distance between nodes.\n
            The value range of parameter is 20 to 200.

            Defaults to 80.
        """
        if 20 <= node_sep <= 200:
            self.options['nodeSep'] = node_sep
        else:
            raise ValueError("The value range of parameter 'node_sep' is 20 to 200.")

    def set_rank_sep(self, rank_sep: int = 80):
        """
        Set distance between layers.\n
        Under horizontal layout, this parameter represents vertical distance between nodes.\n
        Under vertical layout, this parameter represents horizontal distance between nodes.

        Parameters
        ----------
        rank_sep : int, optional
            The distance between layers.\n
            The value range of parameter is 20 to 200.

            Defaults to 80.
        """
        if 20 <= rank_sep <= 200:
            self.options['rankSep'] = rank_sep
        else:
            raise ValueError("The value range of parameter 'rank_sep' is 20 to 200.")


class BaseDigraph(object):
    __TEMPLATE = TemplateUtil.get_template('digraph.html')

    """
    The BaseDigraph is parent class of Digraph and ChildDigraph classes.
    """
    def __init__(self):
        self.template = BaseDigraph.__TEMPLATE

        self.nodes: List[Node] = []
        self.edges: List[Edge] = []

        self.base_node_id: int = 0
        self.lock: Lock = Lock()

    def __add_node(self, name: str, icon_id: int, content: str, in_ports: list, out_ports: list) -> Node:
        self.lock.acquire()
        node_id: int = self.base_node_id + 1
        self.base_node_id = node_id
        self.lock.release()

        added_node: Node = Node(node_id, name, icon_id, content, in_ports, out_ports)
        self.nodes.append(added_node)
        return added_node

    def add_model_node(self, name: str, content: str, in_ports: list, out_ports: list) -> Node:
        """
        Add node with model icon to digraph instance.

        Parameters
        ----------
        name : str
            The model node name.
        content : str
            The model node content.
        in_ports : list
            List of input port names.
        out_ports : list
            List of output port names.

        Returns
        -------
        Node
            The added node with model icon.
        """
        return self.__add_node(name, 0, content, in_ports, out_ports)

    def add_python_node(self, name: str, content: str, in_ports: List, out_ports: List) -> Node:
        """
        Add node with python icon to digraph instance.

        Parameters
        ----------
        name : str
            The python node name.
        content : str
            The python node content.
        in_ports : list
            List of input port names.
        out_ports : list
            List of output port names.

        Returns
        -------
        Node
            The added node with python icon.
        """
        return self.__add_node(name, 1, content, in_ports, out_ports)

    def add_edge(self, source_port: OutPort, target_port: InPort) -> Edge:
        """
        Add edge to digraph instance.

        Parameters
        ----------
        source_port : OutPort
            Start connection point of edge.
        target_port : InPort
            End connection point of edge.

        Returns
        -------
        Edge
            The added edge.
        """
        added_edge = Edge(source_port, target_port)
        self.edges.append(added_edge)
        return added_edge


class Digraph(BaseDigraph):
    """
    Using the Digraph class of digraph framework can dynamically add nodes and edges, and finally generate an HTML page.
    The rendered HTML page can display the node information and the relationship between nodes, and provide a series of auxiliary tools to help you view the digraph.
    A series of auxiliary tools are provided as follows:

    - Provide basic functions such as pan and zoom.
    - Locate the specified node by keyword search.
    - Look at the layout outline of the whole digraph through the minimap.
    - Through the drop-down menu to switch different digraph.
    - The whole page can be displayed in fullscreen.
    - Adjust the distance between nodes and distance between layers dynamically.
    - Provide the function of node expansion and collapse.

    Parameters
    ----------
    digraph_name : str
        The digraph name.

    Examples
    --------
    0. Importing classes of digraph framework

    >>> from hana_ml.visualizers.digraph import Digraph, Node, Edge

    1. Creating a Digraph instance:

    >>> digraph: Digraph = Digraph('Test1')

    2. Adding two nodes to digraph instance, where the node1 has only one output port and the node2 has only one input port:

    >>> node1: Node = digraph.add_model_node('name1', 'content1', in_ports=[], out_ports=['1'])
    >>> node2: Node = digraph.add_python_node('name2', 'content2', in_ports=['1'], out_ports=[])

    3. Adding an edge to digraph instance, where the output port of node1 points to the input port of node2:

    >>> edge1_2: Edge = digraph.add_edge(node1.out_ports[0], node2.in_ports[0])

    4. Create a DigraphConfig instance:

    >>> digraph_config = DigraphConfig()
    >>> digraph_config.set_digraph_layout('vertical')

    5. Generating notebook iframe:

    >>> digraph.build(digraph_config)
    >>> digraph.generate_notebook_iframe(iframe_height=500)

    .. image:: digraph.png

    6. Generating a local HTML file:

    >>> digraph.generate_html('Test1')

    """
    def __init__(self, digraph_name: str):
        super(Digraph, self).__init__()
        self.name: str = digraph_name

        self.html: str = None
        self.frame_src: str = None
        self.frame_html: str = None

    def to_json(self) -> list:
        """
        Return the nodes and edges data of digraph.

        Returns
        -------
        list
            The nodes and edges data of digraph.
        """
        nodes = []
        edges = []

        for node in self.nodes:
            nodes.append(node.json_data)
        for edge in self.edges:
            edges.append(edge.json_data)

        json_data = [{
            'graphId': '1',
            'graphName': self.name,
            'nodes': nodes,
            'edges': edges
        }]
        return json_data

    def build(self, digraph_config: DigraphConfig = None):
        """
        Build HTML string based on current data.

        Parameters
        ----------
        digraph_config : DigraphConfig, optional
            Configuration instance of digraph.
        """
        if digraph_config is None:
            digraph_config = DigraphConfig()

        self.html = self.template.render(
            project_name=self.name,
            layout=digraph_config.options['digraphLayout'],
            nodeSep=digraph_config.options['nodeSep'],
            rankSep=digraph_config.options['rankSep'],
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            data_json=self.to_json(),
            makeTextCenter=digraph_config.options['makeTextCenter'])
        self.html = HTMLUtils.minify(self.html)
        self.frame_src = HTMLFrameUtils.build_frame_src(self.html)

    def generate_html(self, filename: str):
        """
        Save the digraph as a html file.

        Parameters
        ----------
        filename : str
            HTML file name.
        """
        if self.html is None:
            raise Exception(build_html_exception_msg)

        TemplateUtil.generate_html_file('{}_digraph.html'.format(filename), self.html)

    def generate_notebook_iframe(self, iframe_height: int = 800):
        """
        Render the digraph as a notebook iframe.
        """
        if self.html is None:
            raise Exception(build_html_exception_msg)

        HTMLFrameUtils.check_frame_height(iframe_height)

        fullscreen: Fullscreen = Fullscreen('digraph')
        self.frame_html = HTMLFrameUtils.build_frame_html_with_id(fullscreen.target_frame_id, self.frame_src, iframe_height)

        fullscreen.generate_notebook_iframe()
        HTMLFrameUtils.display(self.frame_html)


class MultiDigraph(object):
    """
    Using the MultiDigraph class of digraph framework can dynamically add multiple child digraphs, and finally generate an HTML page.
    The rendered HTML page can display the node information and the relationship between nodes, and provide a series of auxiliary tools to help you view the digraph.
    A series of auxiliary tools are provided as follows:

    - Provide basic functions such as pan and zoom.
    - Locate the specified node by keyword search.
    - Look at the layout outline of the whole digraph through the minimap.
    - Through the drop-down menu to switch different digraph.
    - The whole page can be displayed in fullscreen.
    - Adjust the distance between nodes and distance between layers dynamically.
    - Provide the function of node expansion and collapse.

    Parameters
    ----------
    multi_digraph_name : str
        The digraph name.

    Examples
    --------
    0. Importing classes of digraph framework

    >>> from hana_ml.visualizers.digraph import MultiDigraph, Node, Edge

    1. Creating a MultiDigraph instance:

    >>> multi_digraph: MultiDigraph = MultiDigraph('Test2')

    2. Creating first digraph:

    >>> digraph1 = multi_digraph.add_child_digraph('digraph1')

    3. Adding two nodes to digraph1, where the node1_1 has only one output port and the node2_1 has only one input port:

    >>> node1_1: Node = digraph1.add_model_node('name1', 'content1', in_ports=[], out_ports=['1'])
    >>> node2_1: Node = digraph1.add_python_node('name2', 'content2', in_ports=['1'], out_ports=[])

    4. Adding an edge to digraph1, where the output port of node1_1 points to the input port of node2_1:

    >>> digraph1.add_edge(node1_1.out_ports[0], node2_1.in_ports[0])

    5. Creating second digraph:

    >>> digraph2 = multi_digraph.add_child_digraph('digraph2')

    6. Adding two nodes to digraph2, where the node1_2 has only one output port and the node2_2 has only one input port:

    >>> node1_2: Node = digraph2.add_model_node('name1', 'model text', in_ports=[], out_ports=['1'])
    >>> node2_2: Node = digraph2.add_python_node('name2', 'function info', in_ports=['1'], out_ports=[])

    7. Adding an edge to digraph2, where the output port of node1_2 points to the input port of node2_2:

    >>> digraph2.add_edge(node1_2.out_ports[0], node2_2.in_ports[0])

    8. Generating notebook iframe:

    >>> multi_digraph.build()
    >>> multi_digraph.generate_notebook_iframe(iframe_height=500)

    .. image:: multi_digraph.png

    9. Generating a local HTML file:

    >>> multi_digraph.generate_html('Test2')

    """
    class ChildDigraph(BaseDigraph):
        """
        Multiple child digraphs are logically a whole.
        """
        def __init__(self, child_digraph_id: int, child_digraph_name: str):
            super(MultiDigraph.ChildDigraph, self).__init__()
            if len(child_digraph_name) > 10:
                raise ValueError('The digraph name is longer than ten characters long.')
            self.name: str = child_digraph_name
            self.id: int = child_digraph_id

        def to_json(self) -> list:
            """
            Return the nodes and edges data of child digraph.

            Returns
            -------
            list
                The nodes and edges data of whole digraph.
            """
            nodes = []
            edges = []

            for node in self.nodes:
                nodes.append(node.json_data)
            for edge in self.edges:
                edges.append(edge.json_data)

            json_data = {
                'graphId': str(self.id),
                'graphName': self.name,
                'nodes': nodes,
                'edges': edges
            }
            return json_data

    def __init__(self, multi_digraph_name: str):
        self.name: str = multi_digraph_name

        self.base_child_digraph_id: int = 0
        self.lock: Lock = Lock()
        self.child_digraphs: List[MultiDigraph.ChildDigraph] = []

        self.html: str = None
        self.frame_src: str = None
        self.frame_html: str = None

    def add_child_digraph(self, child_digraph_name: str) -> ChildDigraph:
        """
        Add child digraph to multi_digraph instance.

        Parameters
        ----------
        child_digraph_name : str
            The child digraph name.

        Returns
        -------
        ChildDigraph
            The added child digraph.
        """
        self.lock.acquire()
        child_digraph_id = self.base_child_digraph_id + 1
        self.base_child_digraph_id = child_digraph_id
        self.lock.release()

        added_child_digraph = MultiDigraph.ChildDigraph(child_digraph_id, child_digraph_name)
        self.child_digraphs.append(added_child_digraph)
        return added_child_digraph

    def to_json(self) -> list:
        """
        Return the nodes and edges data of whole digraph.

        Returns
        -------
        list
            The nodes and edges data of whole digraph.
        """
        child_digraph_json = []
        for child_digraph in self.child_digraphs:
            child_digraph_json.append(child_digraph.to_json())
        return child_digraph_json

    def build(self, digraph_config: DigraphConfig = None):
        """
        Build HTML string based on current data.

        Parameters
        ----------
        digraph_config : DigraphConfig, optional
            Configuration instance of digraph.
        """
        if len(self.child_digraphs) == 0:
            raise ValueError('No child digraphs were added!')

        if digraph_config is None:
            digraph_config = DigraphConfig()

        self.html = self.child_digraphs[0].template.render(
            project_name=self.name,
            layout=digraph_config.options['digraphLayout'],
            nodeSep=digraph_config.options['nodeSep'],
            rankSep=digraph_config.options['rankSep'],
            start_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            data_json=self.to_json(),
            makeTextCenter=digraph_config.options['makeTextCenter'])
        self.html = HTMLUtils.minify(self.html)
        self.frame_src = HTMLFrameUtils.build_frame_src(self.html)

    def generate_html(self, filename: str):
        """
        Save the digraph as a html file.

        Parameters
        ----------
        filename : str
            Html file name.
        """
        if self.html is None:
            raise Exception(build_html_exception_msg)

        TemplateUtil.generate_html_file('{}_digraph.html'.format(filename), self.html)

    def generate_notebook_iframe(self, iframe_height: int = 800):
        """
        Render the digraph as a notebook iframe.
        """
        if self.html is None:
            raise Exception(build_html_exception_msg)

        HTMLFrameUtils.check_frame_height(iframe_height)

        fullscreen: Fullscreen = Fullscreen('digraph')
        self.frame_html = HTMLFrameUtils.build_frame_html_with_id(fullscreen.target_frame_id, self.frame_src, iframe_height)

        fullscreen.generate_notebook_iframe()
        HTMLFrameUtils.display(self.frame_html)

