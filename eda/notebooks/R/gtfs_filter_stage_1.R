library(tidytransit)
library(osmdata)

# Load GTFS
gtfs <- read_gtfs("/home/rico/Documents/thesis/eda/notebooks/sample_data/baseline/gtfs.zip")
# Filter for Mondays and filter for rush-hour 7-9
gtfs$calendar_dates$wday <- weekdays(gtfs$calendar_dates$date)
first_monday <- gtfs$calendar_dates[gtfs$calendar_dates$wday=='maandag',]$date[1]
gtfs <- filter_feed_by_date(gtfs, first_monday, "07:00:00", "09:00:00")
gtfs$calendar_dates$wday <- NULL
# Filter for Amsterdam
ams_bbox <- getbb('Amsterdam')
ams_bbox_vec <- c(ams_bbox[1,1], ams_bbox[2,1], ams_bbox[1,2], ams_bbox[2,2]) # xmin, ymin, xmax, ymax
gtfs <- filter_feed_by_area(gtfs, ams_bbox_vec)
# Save GTFS
write_gtfs(gtfs, "/home/rico/Documents/thesis/eda/notebooks/sample_data/baseline/gtfs_reduced_1.zip")
