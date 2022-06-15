from pathlib import Path
import geopandas as gpd
from ptnrue.graph_generation.problem_graph_generator import ProblemGraphGenerator


if __name__ == "__main__":
    AMS_DATA = Path('../ams_data/')

    poi_gdf = gpd.read_file(AMS_DATA.joinpath('non_residential_functions_geojson_latlng.json'))
    poi_gdf = poi_gdf[poi_gdf.Functie == 'Onderwijs']
    poi_gdf.geometry = gpd.points_from_xy(poi_gdf.geometry.y, poi_gdf.geometry.x, crs='EPSG:4326')
    poi_gdf['name'] = poi_gdf['Verblijfsobject']

    census_gdf = gpd.read_parquet(AMS_DATA.joinpath('kwb_21_ams_neighborhoods.parquet'))
    census_gdf = gpd.GeoDataFrame(census_gdf[['BU_NAAM', 'res_centroid']], geometry='res_centroid')
    census_gdf['name'] = census_gdf['BU_NAAM']

    city = "Amsterdam"
    gtfs_zip_file_path = AMS_DATA.joinpath('gtfs.zip')
    out_dir_path = AMS_DATA.joinpath('resulting_graph/')
    day = "monday"
    time_from = "07:00:00"
    time_to = "09:00:00"

    graph_generator = ProblemGraphGenerator(city=city, gtfs_zip_file_path=gtfs_zip_file_path,
                                            out_dir_path=out_dir_path, day=day,
                                            time_from=time_from, time_to=time_to,
                                            agencies=['GVB', 'IFF:GVB', 'IFF:NS', 'IFF:NSI', 'IFF:RNET'],
                                            poi_gdf=poi_gdf, census_gdf=census_gdf)
    graph_generator.generate_problem_graph()
