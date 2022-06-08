library(gtfstools)
library(osmdata)
library(sf)

# Filter by Agencies
gtfs <- read_gtfs('/home/rico/Documents/thesis/eda/notebooks/sample_data/baseline/gtfs_reduced_1.zip')
gtfs <- filter_by_agency_id(gtfs, c('IFF:NS', 'IFF:RNET', 'GVB'))
# Filter for Amsterdam
ams_bbox_poly <- getbb('Amsterdam', format_out = 'sf_polygon')
gtfs <- filter_by_sf(gtfs, ams_bbox_poly$multipolygon)
write_gtfs(gtfs, '/home/rico/Documents/thesis/eda/notebooks/sample_data/baseline/gtfs_reduced_2.zip')
