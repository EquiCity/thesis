import os
import re
import time
from typing import Tuple, List, Union
from pathlib import Path
from zipfile import ZipFile
from multiprocessing.pool import ThreadPool

import networkx as nx
import urbanaccess as ua
from urbanaccess.config import settings
import pandas as pd
import subprocess

from utils.graph_helper_utils import (
    ua_transit_network_to_nx,
    append_length_attribute,
    append_hourly_edge_frequency_attribute,
    append_hourly_stop_frequency_attribute,
)
from utils.osm_utils import get_bbox
from utils.frequency_computation_utils import (
    compute_stop_frequencies,
    compute_segment_frequencies,
)
from exceptions import GraphGenerationError
from utils.file_management_utils import (
    check_or_create_out_dir,
    remove_files_in_dir,
)
import igraph as ig
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GTFSGraphGenerator:

    def __init__(self, city: str, gtfs_zip_file_path: Path, out_dir_path: Path,
                 day: str, time_from: str, time_to: str,
                 agencies: List[str] = None, contract_vertices: bool = False) -> None:
        self.city = city
        bbox = get_bbox(city)
        # (lng_max, lat_min, lng_min, lat_max)
        self.bbox = (bbox['west'], bbox['south'], bbox['east'], bbox['north'])
        self.gtfs_file_path = gtfs_zip_file_path
        self.out_dir_path = out_dir_path
        self.agencies = agencies
        self.day = day
        self.time_from = time_from
        self.time_to = time_to
        self.contract_vertices = contract_vertices

    def _filter_gtfs(self):
        out_path = self.gtfs_file_path.parent\
            .joinpath(f"{self.gtfs_file_path.with_suffix('').name}-filtered-by-{'_'.join(self.agencies)}.zip")

        # Needed to make sure that the agencies that we want to filter for
        # are actually available
        with ZipFile(self.gtfs_file_path, 'r') as gtfs:
            with gtfs.open('agency.txt') as agencies:
                df = pd.read_csv(agencies)
                available_agencies = set(df.agency_id.to_list())

        if not os.path.exists(out_path):
            available_agencies = available_agencies.intersection(set(self.agencies))
            args = [e for sublist in [["-extract-agency", agency] for agency in available_agencies] for e in sublist]
            tool = "transitland"
            subprocess.run([tool, "extract", *args, self.gtfs_file_path, out_path])

        return out_path

    def _loaf_ua_feed(self):
        with ZipFile(self.gtfs_file_path) as ref:
            # First extract all GTFS files as UA cannot load ZIPs directly
            # Could use tmpfile but it's a major pain and the out dir is there anyway
            ref.extractall(self.out_dir_path)
            # Load feed into memory
            loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=str(self.out_dir_path.absolute()),
                                                       validation=True,
                                                       verbose=True,
                                                       bbox=self.bbox,
                                                       remove_stops_outsidebbox=True,
                                                       append_definitions=True)
            # Remove all the extracted files as they are in memory now
            remove_files_in_dir(self.out_dir_path, 'txt')
        return loaded_feeds

    def generate_and_store_graph(self) -> Path:
        # If agencies are provided filter GTFS by them
        if self.agencies:
            self.gtfs_file_path = self._filter_gtfs()

        # Check if file exists already
        with ZipFile(self.gtfs_file_path, 'a') as ref:
            with ref.open('calendar_dates.txt', 'r') as calendar_dates:
                calendar_dates_df = pd.read_csv(calendar_dates)

        # Extract the date from the current GTFS file
        dates = calendar_dates_df['date'].unique()
        gml_file_name = f"{self.city}_pt_network_monday_{dates.min()}_{dates.max()}.gml"
        gml_out_path = self.out_dir_path.joinpath(gml_file_name)

        if not gml_out_path.exists():
            # Load GTFS data
            loaded_feeds = self._loaf_ua_feed()

            # Create the transit network graph from GTFS feeds using the urbanaccess library
            try:
                transit_net = ua.gtfs.network.create_transit_net(
                    gtfsfeeds_dfs=loaded_feeds,
                    calendar_dates_lookup={'unique_feed_id': f"{self.out_dir_path.name}_1"},
                    day=self.day,
                    timerange=[self.time_from, self.time_to],
                )

                # Generate transit graph WITHOUT headways
                G_transit = ua_transit_network_to_nx(transit_net)
                G_transit = append_length_attribute(G_transit)

                # Contract vertices if requested
                if self.contract_vertices:
                    ig_G_transit = ig.Graph.from_networkx(G_transit)
                    g_stop_name_clustering = ig.clustering.VertexClustering.FromAttribute(ig_G_transit, "stop_name")
                    membership = g_stop_name_clustering.membership
                    # Is in-place
                    ig_G_transit.contract_vertices(membership, combine_attrs='first')
                    ig.write(ig_G_transit, gml_out_path)
                else:
                    nx.write_gml(G_transit, gml_out_path)

            except Exception as e:
                logger.error(str(e))
                raise GraphGenerationError(self.out_dir_path)

        return gml_out_path
