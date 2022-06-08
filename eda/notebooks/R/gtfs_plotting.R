library(tidytransit)
library(ggplot2)
library(sf)

# Plot GTFS
gtfs <- read_gtfs('/home/rico/Documents/thesis/eda/notebooks/sample_data/baseline/gtfs_reduced_2.zip')
gtfs_sf <- gtfs_as_sf(gtfs)
routes_sf <- get_route_geometry(gtfs_sf)
# routes_sf_crs <- sf::st_transform(routes_sf, 4326)
routes_sf_buffer <- st_buffer(routes_sf, 0.005)
routes_sf_buffer %>%
  ggplot() +
  geom_sf(colour = alpha("white", 0), fill = alpha("red",0.2)) +
  theme_bw()

# Load new reduced GTFS
# Load Origins from Buurten GEOJSON
# Load Destinations GEOJSON

# Computer OD-Traveltime Matrices

# Save OD-Traveltime Matrices