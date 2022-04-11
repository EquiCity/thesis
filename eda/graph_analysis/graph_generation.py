"""Generates and the walk/bike/transit/car graphs using networkx and osmnx.
"""

# %%
import networkx as nx
import osmnx as ox
import urbanaccess
from zipfile import ZipFile
# from ua2nx import urbanaccess_to_nx
# from library.network_utils import ua_transit_network_to_nx, append_length_attribute, add_transfer_edges


# %%
def create_graph(gdf, network_type, largest_component=False, speed=None):
    G = ox.graph_from_bbox(
        gdf.loc[0, 'bbox_north'],
        gdf.loc[0, 'bbox_south'],
        gdf.loc[0, 'bbox_east'],
        gdf.loc[0, 'bbox_west'],
        network_type)

    if largest_component:
        G = ox.utils_graph.get_largest_component(G, strongly=True)

    if speed:
        nx.set_edge_attributes(G, speed, 'speed_kph')
    else:
        G = ox.speed.add_edge_speeds(G, precision=1)

    G = ox.speed.add_edge_travel_times(G, precision=1)

    return G


# %%
# get the boundaries of Amsterdam
gdf = ox.geocode_to_gdf({'city': 'Amsterdam'})

# create the graphs
# G_walk = create_graph(gdf, 'walk', speed=5)
# G_bike = create_graph(gdf, 'bike', speed=15)
# G_drive = create_graph(gdf, 'drive', largest_component=True)

# %% Create the transit network graph from GTFS feeds using the urbanaccess library

bbox = (gdf.loc[0, 'bbox_west'], gdf.loc[0, 'bbox_south'], gdf.loc[0, 'bbox_east'], gdf.loc[0, 'bbox_north'])
with ZipFile('./data/ov-gtfs-20190105-filtered-by-GVB_IFF:GVB_IFF:NS_IFF:NSI.zip') as zip_ref:
    zip_ref.extractall('/home/rico/Documents/thesis/eda/graph_analysis/tmp_data')
loaded_feeds = urbanaccess.gtfs.load.gtfsfeed_to_df('/home/rico/Documents/thesis/eda/graph_analysis/tmp_data',
                                                    validation=True,
                                                    verbose=True,
                                                    bbox=bbox,
                                                    remove_stops_outsidebbox=True,
                                                    append_definitions=True)

transit_net = urbanaccess.gtfs.network.create_transit_net(
    gtfsfeeds_dfs=loaded_feeds,
    calendar_dates_lookup={'unique_feed_id': 'tmp_data_1'},
    day='monday',
    timerange=['07:00:00', '09:00:00'])

# Note - this needs to be done after create_transit_net because that's when
# stop_times_int is calculated. This means that G_transit does not currently
# contain headway information in its calculations.
loaded_feeds = urbanaccess.gtfs.headways.headways(loaded_feeds,
                                                  headway_timerange=[
                                                      '07:00:00', '09:00:00'])

# Calculate average headways by stop
hw = loaded_feeds.headways.drop_duplicates()
hw['mean_sum'] = hw['count'] * hw['mean']

hw = hw.groupby('unique_stop_id')[['mean_sum', 'count']].sum().reset_index()
hw['mean_hw'] = hw['mean_sum'] / hw[
    'count'] / 2.0  # TODO: figure out - I am not sure why /2 but they also do it in urban access.
hw = hw[['unique_stop_id', 'mean_hw']]

G_transit = ua_transit_network_to_nx(transit_net)
G_transit = append_length_attribute(G_transit)
G_transit = add_transfer_edges(G_transit, hw)

# Add walking network to the transit network.
osm_nodes, osm_edges = urbanaccess.osm.load.ua_network_from_bbox(bbox=bbox)

ua_network = urbanaccess.osm.network.create_osm_net(
    osm_edges=osm_edges,
    osm_nodes=osm_nodes,
    travel_speed_mph=3,
    network_type='walk')

urbanaccess_nw = urbanaccess.network.integrate_network(
    urbanaccess_network=ua_network,
    urbanaccess_gtfsfeeds_df=loaded_feeds,
    headways=True,
    headway_statistic='mean'
)

G_transit_walk = urbanaccess_to_nx(urbanaccess_nw, optimize='median')
G_transit_walk = append_length_attribute(G_transit_walk)

# %% Save Graphs
# nx.write_gpickle(G_drive, './graphs/ams_drive_network.gpickle')
# nx.write_gpickle(G_bike, './graphs/ams_bike_network.gpickle')
# nx.write_gpickle(G_walk, './graphs/ams_walk_network.gpickle')

nx.write_gpickle(G_transit, './graphs/ams_transit_network_transfer_hw_correct.gpickle')
nx.write_gpickle(G_transit_walk, './graphs/ams_transit_walk_network.gpickle')

# GML files are used to read networks into igraph
nx.write_gml(G_transit, './graphs/ams_transit_network_transfer_hw_correct.gml')
nx.write_gml(G_transit_walk, './graphs/ams_transit_walk_network.gml')

# %% Print Graphs
# fig, ax = ox.plot_graph(G_drive, node_size=3)
# fig.savefig('./images/amsterdam_drive_network.png')

# fig, ax = ox.plot_graph(G_bike, node_size=3)
# fig.savefig('./images/amsterdam_bike_network.png')

# fig, ax = ox.plot_graph(G_walk, node_size=3)
# fig.savefig('./images/amsterdam_walk_network.png')

# fig, ax = ox.plot_graph(G_transit, node_size=3)
# fig.savefig('./images/amsterdam_transit_network.png')

# For node colouring
# nc = ['r' if node in (2094064399, 46408191) else 'w' for node in G.nodes()]

# %% Deprecated: transit network using peartree
# feed = pt.get_representative_feed('./input/gtfs.zip')
# start = 7 * 60 * 60
# end = 9 * 60 * 60
# # Peartree networks generated from GTFS data already calculate edge lengths in seconds, so we do not need to add travel times, etc.
# G_transit = pt.load_feed_as_graph(feed, start, end)
# # Truncate the network to the amsterdam area
# G_transit = ox.truncate.truncate_graph_bbox(G_transit, gdf.loc[0, 'bbox_north'], gdf.loc[0, 'bbox_south'], gdf.loc[0, 'bbox_east'], gdf.loc[0, 'bbox_west'])
