from datetime import datetime
import itertools as it

import geopandas as gpd
from typing import List
import igraph as ig

import numpy
import osmnx as ox
import numpy as np
import networkx as nx
from pathlib import Path
from experiments.constants.osm_network_types import OSMNetworkTypes
from gtfs_graph_generator import GTFSGraphGenerator
from osm_graph_generation import OSMGraphGenerator
from utils.list_utils import check_all_lists_of_same_length
from experiments.constants.travel_speed import MetricTravelSpeed
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _add_points_to_graph(g: ig.Graph, xs: List[float], ys: List[float],
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

    # Really don't want to set name attribute
    if 'name' in kwargs:
        del kwargs['name']

    # Add points as vertices
    v_attrs = {
        'x': xs,
        'y': ys,
        'type': v_type,
        'color': color,
    }
    v_attrs.update(kwargs)
    # Is in-place
    g.add_vertices(len(xs), v_attrs)


def _add_edges_to_graph(g: ig.Graph, osm_graph: nx.MultiDiGraph, from_node_type: str, to_node_type: str,
                        e_type: str, speed: float, color: str = None) -> None:
    """

    Args:
        g:
        from_nodes:
        to_nodes:
        distances:
        e_type:
        speed:
        color:

    Returns:

    """
    from_nodes = g.vs.select(type_eq=from_node_type)
    to_nodes = g.vs.select(type_eq=to_node_type)

    edges = list(it.product(from_nodes, to_nodes))
    edges_from = np.array([[e[0]['x'], e[0]['y']] for e in edges])
    edges_to = np.array([[e[1]['x'], e[1]['y']] for e in edges])

    distances = []

    orig_nodes = ox.distance.nearest_nodes(osm_graph, edges_from[:, 0], edges_from[:, 1])
    # dest_nodes = ox.distance.nearest_nodes(osm_graph, edges_to[:, 0], edges_to[:, 1])
    # osmnx_routes = ox.distance.shortest_path(osm_graph, orig_nodes, dest_nodes, 'length', cpus=None)
    #
    # for route in osmnx_routes:
    #     edge_lengths = ox.utils_graph.get_route_edge_attributes(osm_graph, route, 'length')
    #     route_len_m = sum(edge_lengths)
    #     distances.append(route_len_m)
    #
    # distances = numpy.array(distances)

    # For debugging comment above and uncomment below
    distances = numpy.array([1000] * len(orig_nodes))

    E_WALK_attr = {
        'distance': distances,
        'type': e_type,
        'tt': (distances / speed) * 60,
        'weight': (distances / speed) * 60,
        'color': color,
    }
    g.add_edges(edges, E_WALK_attr)


class ProblemGraphGenerator:

    def __init__(self, city: str, gtfs_zip_file_path: Path, out_dir_path: Path,
                 day: str, time_from: str, time_to: str, agencies: List[str],
                 poi_gdf: gpd.GeoDataFrame, census_gdf: gpd.GeoDataFrame) -> None:
        """

        Args:
            city:
            gtfs_zip_file_path:
            out_dir_path:
            day:
            time_from:
            time_to:
            poi_gdf:
            census_gdf:
        """
        self.city = city
        self.gtfs_graph_generator = GTFSGraphGenerator(city=city, gtfs_zip_file_path=gtfs_zip_file_path,
                                                       out_dir_path=out_dir_path, day=day,
                                                       time_from=time_from, time_to=time_to, agencies=agencies,
                                                       contract_vertices=True)
        self.osm_graph_generator = OSMGraphGenerator(city=city, network_type=OSMNetworkTypes.WALK,
                                                     graph_out_path=out_dir_path)
        self.out_dir_path = out_dir_path
        self.poi_gdf = poi_gdf
        self.census_gdf = census_gdf

    def generate_problem_graph(self) -> Path:
        """

        Returns:

        """

        # Generate GTFS Graph
        logger.debug("Starting GTFS graph generation")
        gtfs_graph_file_path = self.gtfs_graph_generator.generate_and_store_graph()
        logger.debug(f"Created GTFS graph and stored in {gtfs_graph_file_path}")

        # Load GTFS Graph
        logger.debug("Loading GTFS graph")
        pt_graph = ig.read(gtfs_graph_file_path)

        # Generate OSM Graph
        logger.debug("Starting OSM graph generation")
        osm_graph_file_path = self.osm_graph_generator.generate_and_store_graph()
        logger.debug(f"Created OSM Graph and stored in {osm_graph_file_path}")
        # Load OSM Graph
        logger.debug("Loading OSM graph")
        osm_graph = nx.read_gpickle(osm_graph_file_path)

        # Build new graph starting from the GTFS graph
        logger.debug("###\nStarting problem graph generation")
        g: ig.Graph = pt_graph.copy()
        # Set all existing vertices to be of type public transport
        g.vs.set_attribute_values(attrname='type', values='pt_node')

        # Add all residential centroids as vertices
        logger.debug("Adding residential centroid vertices to graph")
        rc_names = self.census_gdf.name.to_list()
        rc_xs = self.census_gdf.geometry.x.to_numpy()
        rc_ys = self.census_gdf.geometry.y.to_numpy()

        if not len(set(rc_names)) == len(rc_names):
            raise ValueError("Names of residential centroids in the GeoDataFrames have to be unique")

        # Names have to be integers!
        _add_points_to_graph(g=g, xs=rc_xs, ys=rc_ys, v_type='res_node', color='red', ref_name=rc_names)

        # Add all POIs as vertices
        logger.debug("Adding POI vertices to graph")
        poi_names = self.poi_gdf.name.to_list()
        poi_xs = self.poi_gdf.geometry.x.to_numpy()
        poi_ys = self.poi_gdf.geometry.y.to_numpy()

        if not len(set(poi_names)) == len(poi_names):
            raise ValueError("Names of POIs in the GeoDataFrames have to be unique")

        _add_points_to_graph(g=g, xs=poi_xs, ys=poi_ys, v_type='poi_node', color='green', ref_name=poi_names)

        # Add edges from all res centroids to all POIs
        logger.debug(f"Adding edges res_node->poi_node")
        _add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='res_node', to_node_type='poi_node',
                            e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='grey')

        # Add edges from all res centroids to all PT stations
        logger.debug(f"Adding edges res_node->pt_node")
        _add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='res_node', to_node_type='pt_node',
                            e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='grey')

        # Add edges from all PT stations to all POIs
        logger.debug(f"Adding edges pt_node->poi_node")
        _add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='pt_node', to_node_type='poi_node',
                            e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='grey')

        # Set all edges to be active
        g.es['active'] = 1
        # Is needed to not generate an exception
        del g.vs['id']

        logger.debug("Writing final problem graph")
        final_out_file = self.out_dir_path.joinpath(f"{self.city}_problem_graph_{datetime.now().date()}.gml")
        ig.write(g, final_out_file)

        return final_out_file
