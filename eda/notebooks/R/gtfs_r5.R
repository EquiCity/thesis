options(java.parameters = '-Xmx2G')
library(r5r)
library(sf)
library(data.table)
library(ggplot2)
library(mapview)
mapviewOptions(platform = 'leafgl')

data_path <- file.path("/home/rico/Documents/thesis/eda/notebooks/sample_setup")

# Clean up
old_files <- c(file.path(data_path,'network.dat'),
               file.path(data_path,'osm.pbf.mapdb'),
               file.path(data_path,'osm.pbf.mapdb.p'))
for (file in old_files) {
  if (file.exists(file)) {
    file.remove(file)
  }
}

poi <- fread(file.path(data_path, "work_points.csv"))
points <- fread(file.path(data_path, "hh_points.csv"))

# Indicate the path where OSM and GTFS data are stored
r5r_core <- setup_r5(data_path = data_path, verbose = TRUE)

# set inputs
mode <- c("WALK", "TRANSIT")
max_walk_dist <- 1000
max_trip_duration <- 500
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
                          verbose = TRUE)

# calculate detailed itineraries
dit <- detailed_itineraries(r5r_core = r5r_core,
                            origins = points[263:263+263],
                            destinations = poi,
                            mode = mode,
                            departure_datetime = departure_datetime,
                            max_walk_dist = max_walk_dist,
                            shortest_path = TRUE,
                            verbose = TRUE)

write.csv2(ttm, file.path(data_path, "travel_times.csv"), row.names=FALSE)

stop_r5(r5r_core)
rJava::.jgc(R.gc = TRUE)
