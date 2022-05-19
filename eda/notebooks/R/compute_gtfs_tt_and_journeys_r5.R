options(java.parameters = '-Xmx2G')
library(r5r)
library(sf)
library(data.table)
library(ggplot2)
library(mapview)
mapviewOptions(platform = 'leafgl')

data_path <- file.path("/home/rico/Documents/thesis/eda/notebooks/sample_setup")
poi <- fread(file.path(data_path, "work_points.csv"))
points <- fread(file.path(data_path, "hh_points.csv"))

# Indicate the path where OSM and GTFS data are stored
r5r_core <- setup_r5(data_path = data_path, verbose = TRUE)

# set inputs
mode <- c("TRANSIT")
max_walk_dist <- 50
max_trip_duration <- 120
departure_datetime <- as.POSIXct("11-11-2019 07:00:00",
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


# calculate detailed itineraries
dit <- detailed_itineraries(r5r_core = r5r_core,
                            origins = points[250:500],
                            destinations = poi,
                            mode = mode,
                            departure_datetime = departure_datetime,
                            max_walk_dist = max_walk_dist,
                            shortest_path = TRUE,
                            verbose = FALSE)

write.csv2(ttm, file.path(data_path, "travel_times.csv"), row.names=FALSE)

stop_r5(r5r_core)
rJava::.jgc(R.gc = TRUE)
