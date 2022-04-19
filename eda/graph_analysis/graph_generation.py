import os
import re
import time
from typing import Tuple, List
from pathlib import Path
from zipfile import ZipFile
from functools import cache
from multiprocessing.pool import ThreadPool

import networkx as nx
import urbanaccess as ua
from ua2nx import urbanaccess_to_nx as ua_to_nx
from urbanaccess.config import settings

from graph_helper_utils import (
    ua_transit_network_to_nx,
    add_transfer_edges,
    append_length_attribute,
)
from utils.osm_utils import get_bbox
from osm_network_types import OSMNetworkTypes
from speeds import ImperialTravelSpeeds

import logging

# Logger and output config
settings.log_consolse = False
logging.basicConfig()
logger = logging.getLogger('graph_extraction')
logger.setLevel(logging.DEBUG)

# Settings
NUM_WORKERS = os.getenv('NUM_WORKERS', 4)

# Data paths
GTFS_DATA_DIR = Path(os.getenv('GTFS_DATA_DIR', './data/day_gtfs_files'))
TRANSIT_GRAPH_DATA_DIR = Path(os.getenv('TRANSIT_GRAPH_DATA_DIR', './data/transit_graphs'))


@cache
def _get_osm_nodes_and_edges(bbox: Tuple):
    return ua.osm.load.ua_network_from_bbox(bbox=bbox)


def _generate_and_store_graphs(args: Tuple[dict, Path]) -> Path:
    logger.debug(f"received: {args}")
    bbox, gtfs_file = args

    curr_run_dir = TRANSIT_GRAPH_DATA_DIR.joinpath(gtfs_file.with_suffix('').name)
    if os.path.exists(curr_run_dir):
        logger.warning(f"Directory {curr_run_dir} already exists -> removing.")
        return curr_run_dir
        # for file in os.listdir(curr_run_dir):
        #     os.remove(curr_run_dir.joinpath(file))
        # os.rmdir(curr_run_dir)
    else:
        os.mkdir(curr_run_dir)

    with ZipFile(gtfs_file) as ref:
        ref.extractall(curr_run_dir)
        loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=str(curr_run_dir.absolute()),
                                                   validation=True,
                                                   verbose=True,
                                                   bbox=bbox,
                                                   remove_stops_outsidebbox=True,
                                                   append_definitions=True)
        for f in os.listdir(curr_run_dir):
            os.remove(curr_run_dir.joinpath(f))

    # Create the transit network graph from GTFS feeds using the urbanaccess library
    logger.debug(loaded_feeds.calendar_dates.columns)
    transit_net = ua.gtfs.network.create_transit_net(
        gtfsfeeds_dfs=loaded_feeds,
        calendar_dates_lookup={'unique_feed_id': f"{gtfs_file.with_suffix('').name}_1"},
        day='monday',
        timerange=['07:00:00', '09:00:00'],
    )
    # Note - this needs to be done after create_transit_net because that's when
    # stop_times_int is calculated. This means that G_transit does not currently
    # contain headway information in its calculations.
    loaded_feeds = ua.gtfs.headways.headways(loaded_feeds,
                                             headway_timerange=['07:00:00', '09:00:00'])

    # Calculate average headways by stop
    hw = loaded_feeds.headways.drop_duplicates()
    hw['mean_sum'] = hw['count'] * hw['mean']

    hw = hw.groupby('unique_stop_id')[['mean_sum', 'count']].sum().reset_index()
    # TODO: figure out - I am not sure why /2 but they also do it in urban access.
    hw['mean_hw'] = hw['mean_sum'] / hw['count'] / 2.0
    hw = hw[['unique_stop_id', 'mean_hw']]

    # Generate transit graph
    G_transit = ua_transit_network_to_nx(transit_net)
    G_transit = append_length_attribute(G_transit)
    G_transit = add_transfer_edges(G_transit, hw)

    osm_nodes, osm_edges = _get_osm_nodes_and_edges(bbox)

    ua_network = ua.osm.network.create_osm_net(
        osm_edges=osm_edges,
        osm_nodes=osm_nodes,
        travel_speed_mph=ImperialTravelSpeeds.WALKING.value,
        network_type=OSMNetworkTypes.WALK.value)

    urbanaccess_nw = ua.network.integrate_network(
        urbanaccess_network=ua_network,
        urbanaccess_gtfsfeeds_df=loaded_feeds,
        headways=True,
        headway_statistic='mean'
    )

    G_transit_walk = ua_to_nx(urbanaccess_nw, optimize='median')
    G_transit_walk = append_length_attribute(G_transit_walk)

    # Extract the date from the current GTFS file
    date = re.findall(r'\d+', str(gtfs_file))[0]

    nx.write_gpickle(G_transit_walk, curr_run_dir.joinpath(f'ams_transit_walk_network_{date}.gpickle'))
    nx.write_gml(G_transit_walk, curr_run_dir.joinpath(f'ams_transit_walk_network_{date}.gml'))

    nx.write_gpickle(G_transit, curr_run_dir.joinpath(f'ams_transit_network_transfer_hw_correct_{date}.gpickle'))
    nx.write_gml(G_transit, curr_run_dir.joinpath(f'ams_transit_network_transfer_hw_correct_{date}.gml'))

    return curr_run_dir


def generate_transit_graphs(bbox_dict: dict, gtfs_day_files: List[Path]):
    # (lng_max, lat_min, lng_min, lat_max)
    bbox = (
        bbox_dict['west'],
        bbox_dict['south'],
        bbox_dict['east'],
        bbox_dict['north'],
    )

    stored_graphs = []
    total_space = 0
    start = time.time()

    inputs = [[bbox, gtfs_day_file] for gtfs_day_file in gtfs_day_files]
    logger.debug(inputs)
    results = ThreadPool(NUM_WORKERS).imap_unordered(_generate_and_store_graphs, inputs)

    for r in results:
        stored_graphs.append(r)
        total_space += r.stat().st_size

    logger.info(f"###\n"
                f"Processed {len(stored_graphs)} graphs\n"
                f"Amounting to {total_space / (1024.0 * 1024.0)} MB\n"
                f"Took {time.time()-start} seconds\n"
                f"###")


if __name__ == "__main__":
    # Aggregate needed data
    bbox_dict = get_bbox('Amsterdam')
    all_gtfs_files = [GTFS_DATA_DIR.joinpath(e) for e in os.listdir(GTFS_DATA_DIR) if Path(e).suffix == '.zip']

    # Run the core part
    generate_transit_graphs(bbox_dict, all_gtfs_files)
