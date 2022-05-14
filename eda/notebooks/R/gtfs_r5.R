options(java.parameters = '-Xmx2G')
library(r5r)
library(sf)
library(data.table)
library(ggplot2)
library(mapview)
mapviewOptions(platform = 'leafgl')

data_path <- file.path("/home/rico/Documents/thesis/eda/notebooks/sample_setup")
poi <- fread(file.path(data_path, "ams-nhs-fixed-centroids.csv"))
points <- fread(file.path(data_path, "education_pois_latlng.csv"))

# Indicate the path where OSM and GTFS data are stored
r5r_core <- setup_r5(data_path = data_path, verbose = FALSE)

# set inputs
mode <- c("WALK", "TRANSIT")
max_walk_dist <- 5000
max_trip_duration <- 120
departure_datetime <- as.POSIXct("13-05-2019 14:00:00",
                                 format = "%d-%m-%Y %H:%M:%S")

# calculate a travel time matrix
ttm <- travel_time_matrix(r5r_core = r5r_core,
                          origins = points,
                          destinations = poi,
                          mode = mode,
                          departure_datetime = departure_datetime,
                          max_walk_dist = max_walk_dist,
                          max_trip_duration = max_trip_duration,
                          verbose = FALSE)

head(ttm)

stop_r5(r5r_core)
rJava::.jgc(R.gc = TRUE)
