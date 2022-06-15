import igraph as ig
from typing import List
import numpy as np
import osmnx as ox
import networkx as nx
import itertools as it
from .list_utils import check_all_lists_of_same_length


def add_points_to_graph(g: ig.Graph, names: List[str], xs: List[float], ys: List[float],
                        v_type: str, color: str = None, **kwargs) -> None:
    """

    Args:
        g:
        names:
        xs:
        ys:
        v_type:
        color:

    Returns:

    """
    # Check that all lists are of the same size
    lists = [xs, ys]
    for key, value in kwargs.items():
        lists.append(value)
    check_all_lists_of_same_length(*lists)

    # Add points as vertices
    v_attrs = {
        'name': names,
        'x': xs,
        'y': ys,
        'type': v_type,
        'color': color,
    }
    v_attrs.update(kwargs)
    # Is in-place
    g.add_vertices(len(xs), v_attrs)


def add_edges_to_graph(g: ig.Graph, osm_graph: nx.MultiDiGraph, from_node_type: str, to_node_type: str,
                       e_type: str, speed: float, color: str = None) -> None:
    """

    Args:
        g:
        from_nodes:
        to_nodes:
        distances:
        e_type:
        speed: transit speed in km/h
        color:

    Returns:

    """
    from_nodes = g.vs.select(type_eq=from_node_type)
    to_nodes = g.vs.select(type_eq=to_node_type)

    edges = list(it.product(from_nodes, to_nodes))
    edges_from = np.array([[e[0]['x'], e[0]['y']] for e in edges])
    edges_to = np.array([[e[1]['x'], e[1]['y']] for e in edges])

    distances = []

    # orig_nodes = ox.distance.nearest_nodes(osm_graph, edges_from[:, 0], edges_from[:, 1])
    # dest_nodes = ox.distance.nearest_nodes(osm_graph, edges_to[:, 0], edges_to[:, 1])
    # osmnx_routes = ox.distance.shortest_path(osm_graph, orig_nodes, dest_nodes, 'length', cpus=None)
    #
    # for route in osmnx_routes:
    #     edge_lengths = ox.utils_graph.get_route_edge_attributes(osm_graph, route, 'length')
    #     route_len_m = sum(edge_lengths)
    #     distances.append(route_len_m)

    # For debugging comment above and uncomment below
    distances = [1000] * len(edges)

    distances = np.array(distances)

    edge_attrs = {
        'distance': distances,
        'type': e_type,
        'traveltime': (distances / speed * 1000) * 60,
        'weight': (distances / speed * 1000) * 60,
        'color': color,
    }

    g.add_edges(edges, edge_attrs)
