from datetime import datetime

import geopandas as gpd
from typing import List
import igraph as ig

import networkx as nx
from pathlib import Path
from ..constants.osm_network_types import OSMNetworkTypes
from .gtfs_graph_generator import GTFSGraphGenerator
from .osm_graph_generation import OSMGraphGenerator
from ..constants.travel_speed import MetricTravelSpeed
from .utils.graph_expansion import add_points_to_graph, add_edges_to_graph
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
        add_points_to_graph(g=g, xs=rc_xs, ys=rc_ys, v_type='res_node', color='red', ref_name=rc_names)

        # Add all POIs as vertices
        logger.debug("Adding POI vertices to graph")
        poi_names = self.poi_gdf.name.to_list()
        poi_xs = self.poi_gdf.geometry.x.to_numpy()
        poi_ys = self.poi_gdf.geometry.y.to_numpy()

        if not len(set(poi_names)) == len(poi_names):
            raise ValueError("Names of POIs in the GeoDataFrames have to be unique")

        add_points_to_graph(g=g, xs=poi_xs, ys=poi_ys, v_type='poi_node', color='green', ref_name=poi_names)

        # Add edges from all res centroids to all POIs
        logger.debug(f"Adding edges res_node->poi_node")
        add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='res_node', to_node_type='poi_node',
                            e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='grey')

        # Add edges from all res centroids to all PT stations
        logger.debug(f"Adding edges res_node->pt_node")
        add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='res_node', to_node_type='pt_node',
                            e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='grey')

        # Add edges from all PT stations to all POIs
        logger.debug(f"Adding edges pt_node->poi_node")
        add_edges_to_graph(g=g, osm_graph=osm_graph, from_node_type='pt_node', to_node_type='poi_node',
                            e_type='walk', speed=MetricTravelSpeed.WALKING.value, color='grey')

        # Set all edges to be active
        g.es['active'] = 1
        # Is needed to not generate an exception
        del g.vs['id']

        logger.debug("Writing final problem graph")
        final_out_file = self.out_dir_path.joinpath(f"{self.city}_problem_graph_{datetime.now().date()}.gml")
        ig.write(g, final_out_file)

        return final_out_file