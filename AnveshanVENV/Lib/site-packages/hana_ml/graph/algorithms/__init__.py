"""
This package contains various algorithms you can use to explore
and work on a graph.

The general pattern is: Create an algorithm object instance (which expects
a `Graph` object in the constructor) and then call `execute(<parameters>)`
on that algorithm instance. This can be combined in one statement.

>>> import hana_ml.graph.algorithms as hga
>>> sp = hga.ShortestPath(graph=g).execute(source="1", target="3")

The execute statement always returns the algorithm instance itself, so
that it can used to access the result properties.

>>> print("Vertices", sp.vertices)
>>> print("Edges", sp.edges)
>>> print("Weight:", sp.weight)

If you want to create a new algorithm, have a closer look at
:class:`algorithm_base.AlgorithmBase`.

The following algorithms are available:

    * :class:`KShortestPaths`
    * :class:`Neighbors`
    * :class:`NeighborsSubgraph`
    * :class:`ShortestPath`
    * :class:`ShortestPathsOneToAll`
    * :class:`StronglyConnectedComponents`
    * :class:`WeaklyConnectedComponents`
    * :class:`TopologicalSort`

"""

from .shortest_path import ShortestPath
from .neighbors import Neighbors, NeighborsSubgraph
from .k_shortest_paths import KShortestPaths
from .topo_sort import TopologicalSort
from .shortest_paths_one_to_all import ShortestPathsOneToAll
from .strongly_connected_components import StronglyConnectedComponents
from .weakly_connected_components import WeaklyConnectedComponents
