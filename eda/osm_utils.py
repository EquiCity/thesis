from typing import Tuple
import osmnx as ox


def get_bbox(city: str) -> Tuple[float, float, float, float]:
    gdf = ox.geocode_to_gdf({'city': city})
    return gdf.loc[0, 'bbox_west'], gdf.loc[0, 'bbox_south'], gdf.loc[0, 'bbox_east'], gdf.loc[0, 'bbox_north']
