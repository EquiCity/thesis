import os
from pathlib import Path
import geopandas as gpd
from eptnr.graph_generation.problem_graph_generator import ProblemGraphGenerator

BASE_PATH = Path(os.environ['BASE_DATA_PATH'])


if __name__ == "__main__":
    poi_gdf = gpd.read_file(BASE_PATH.joinpath('poi_latlng.json'))
    poi_gdf = poi_gdf[poi_gdf.Functie == 'Museum en galerie']
    poi_gdf.geometry = gpd.points_from_xy(poi_gdf.geometry.y, poi_gdf.geometry.x, crs='EPSG:4326')
    poi_gdf['name'] = poi_gdf['Verblijfsobject']

    census_gdf = gpd.read_parquet(BASE_PATH.joinpath('neighborhoods.parquet'))
    census_gdf = gpd.GeoDataFrame(census_gdf[['BU_NAAM', 'res_centroid']], geometry='res_centroid')
    census_gdf['name'] = census_gdf['BU_NAAM']

    city = "Amsterdam"
    gtfs_zip_file_path = BASE_PATH.joinpath('gtfs.zip')
    out_dir_path = BASE_PATH.joinpath('resulting_graph/')

    if not os.path.exists(out_dir_path):
        os.mkdir(out_dir_path)

    day = "monday"
    time_from = "07:00:00"
    time_to = "09:00:00"

    modalities = ['metro']

    graph_generator = ProblemGraphGenerator(city=city, gtfs_zip_file_path=gtfs_zip_file_path,
                                            out_dir_path=out_dir_path, day=day,
                                            time_from=time_from, time_to=time_to,
                                            agencies=['GVB', 'IFF:GVB', 'IFF:NS', 'IFF:NSI', 'IFF:RNET'],
                                            poi_gdf=poi_gdf, census_gdf=census_gdf, modalities=modalities,
                                            distances_computation_mode='haversine')

    graph_generator.generate_problem_graph()
