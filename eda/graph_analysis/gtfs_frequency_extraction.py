import urbanaccess as ua
from pathlib import Path
import os
from zipfile import ZipFile
import osmnx as ox

import datetime

import numpy as np
import pandas as pd

GTFS_DATA_DIR = Path('./sample_data/gtfs_monday_extracted')
TMP_DATA_PATH = Path('./tmp')

all_gtfs_files = [GTFS_DATA_DIR.joinpath(e) for e in os.listdir(GTFS_DATA_DIR) if Path(e).suffix == '.zip']
path = all_gtfs_files[0]

gdf = ox.geocode_to_gdf({'city': 'Amsterdam'})
bbox = (
    gdf.loc[0, 'bbox_west'],
    gdf.loc[0, 'bbox_south'],
    gdf.loc[0, 'bbox_east'],
    gdf.loc[0, 'bbox_north'],
)


with ZipFile(path) as ref:
    ref.extractall(TMP_DATA_PATH)
    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(gtfsfeed_path=str(TMP_DATA_PATH.absolute()),
                                               validation=True,
                                               verbose=True,
                                               bbox=bbox,
                                               remove_stops_outsidebbox=True,
                                               append_definitions=True)
    for f in os.listdir(TMP_DATA_PATH):
        os.remove(TMP_DATA_PATH.joinpath(f))


rect_arrivals = loaded_feeds.stop_times['arrival_time'].apply(lambda x: int(x.split(':')[0]))
rect_departures = loaded_feeds.stop_times['departure_time'].apply(lambda x: int(x.split(':')[0]))

# Drop all runs where the arrival time is after midnight
loaded_feeds.stop_times = loaded_feeds.stop_times[rect_arrivals < 24]
loaded_feeds.stop_times = loaded_feeds.stop_times[rect_departures < 24]

date = datetime.datetime.strptime(str(loaded_feeds.calendar_dates.date.unique()[0]), '%Y%m%d')

loaded_feeds.stop_times['arrival_time'] = pd.to_datetime(
    loaded_feeds.stop_times['arrival_time'].apply(lambda x: str(date.date()) + ' ' + x))
loaded_feeds.stop_times['departure_time'] = pd.to_datetime(
    loaded_feeds.stop_times['departure_time'].apply(lambda x: str(date.date()) + ' ' + x))

day_times = pd.to_datetime(pd.Series([date + datetime.timedelta(hours=e) for e in range(25)]))

# ## Stop frequencies
stop_freq = loaded_feeds.stops[["stop_id", "stop_name", "stop_lat", "stop_lon"]]
stop_freq = stop_freq.drop_duplicates()

for h in range(24):
    stop_freq[f"freq_h_{h}"] = np.zeros(len(stop_freq))

for i in range(len(day_times[:24])):
    served_stops = loaded_feeds.stop_times[(loaded_feeds.stop_times.arrival_time >= day_times[i]) & (
            loaded_feeds.stop_times.arrival_time <= day_times[i + 1])]
    served_stops_count = served_stops.groupby('stop_id').size()
    served_stops_count_ids = served_stops_count.index
    stop_freq.loc[stop_freq['stop_id'].isin(served_stops_count_ids), f'freq_h_{i}'] = served_stops_count.values

# ## Segment Frequencies
# Generate arrival_stop_id for each trip
for trip_id in loaded_feeds.stop_times['trip_id'].unique():
    loaded_feeds.stop_times.loc[loaded_feeds.stop_times['trip_id'] == trip_id, "stop_id_provenance"] = \
        loaded_feeds.stop_times.loc[loaded_feeds.stop_times['trip_id'] == trip_id, "stop_id"].shift(1)

seg_freq = loaded_feeds.stop_times[["stop_id", "stop_id_provenance"]]
seg_freq = seg_freq.dropna()
seg_freq = seg_freq.drop_duplicates()
seg_freq.set_index(["stop_id", "stop_id_provenance"], inplace=True)

for h in range(24):
    seg_freq[f"freq_h_{h}"] = np.zeros(len(seg_freq))

for i in range(len(day_times[:24])):
    served_stops = loaded_feeds.stop_times[(loaded_feeds.stop_times.arrival_time >= day_times[i]) & (
            loaded_feeds.stop_times.arrival_time <= day_times[i + 1])]
    serv_counts = served_stops.groupby(["stop_id", "stop_id_provenance"]).size()
    seg_freq.loc[serv_counts.index, f"freq_h_{i}"] = serv_counts.values
