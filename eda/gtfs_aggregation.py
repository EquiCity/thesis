import os
from pathlib import Path
from typing import Optional, List, Tuple, Union
import requests
from multiprocessing.pool import ThreadPool
import datetime
import urbanaccess as ua
import logging
import zipfile

from osm_utils import get_bbox
from transit_feed_providers import TransitFeedProviders
from exceptions import GTFSDownloadException

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


BBOX = get_bbox(city="Amsterdam")
DATA_PATH = './data'


def _fetch_url_and_process_gtfs(entry: Tuple[Path, str]) -> Union[Path, GTFSDownloadException]:
    path, uri = entry

    # Download GTFS if it does not exist
    if not os.path.exists(path):
        r = requests.get(uri, stream=True)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
        else:
            return GTFSDownloadException(f"unable to download {uri}")

    # Extract data
    unzipped_gtfs_path = path.with_suffix('')
    logger.info(f"Unzipping {uri} to {unzipped_gtfs_path}")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(unzipped_gtfs_path)

    # Load GTFS
    loaded_feeds = ua.gtfs.load.gtfsfeed_to_df(str(unzipped_gtfs_path.absolute()),
                                               validation=True,
                                               verbose=True,
                                               bbox=BBOX,
                                               remove_stops_outsidebbox=True,
                                               append_definitions=True)
    # Store as HDF5
    h5_path = path.with_suffix('.h5')
    ua.gtfs.network.save_processed_gtfs_data(loaded_feeds, str(h5_path))
    # Remove GTFS ZIP
    os.remove(path)
    return path


def _create_gtfs_urls(provider: TransitFeedProviders,
                     start_date: datetime.date,
                     end_date: Optional[datetime.date] = None) -> List[Tuple[Path, str]]:
    """

    :param provider:
    :param start_date:
    :param end_date:
    :return:
    """
    if not end_date:
        end_date = datetime.date.today()

    if not start_date < end_date:
        raise Exception("start_date cannot be before or on end_date; it has to be further back in time")

    def perdelta(start: datetime.date, end: datetime.date, delta: datetime.timedelta = datetime.timedelta(days=1)):
        curr = start
        while curr < end:
            yield curr
            curr += delta

    day_dates = perdelta(start_date, end_date, )

    return list(map(lambda x: (Path(f"{DATA_PATH}/ov-gtfs-{x.isoformat().replace('-', '')}.zip"),
                               f"https://transitfeeds.com/p/{provider.value}/{x.isoformat().replace('-', '')}/download"),
                    day_dates)
                )


def download_and_store_gtfs(start_date: datetime.date, end_date: datetime.date, provider: TransitFeedProviders):
    urls = _create_gtfs_urls(provider, start_date, end_date)

    downloaded_paths = []
    total_space = 0

    results = ThreadPool(8).imap_unordered(_fetch_url_and_process_gtfs, urls)
    for path_or_exception in results:
        if not isinstance(path_or_exception, GTFSDownloadException):
            path = path_or_exception
            logger.info(f"downloaded and processed {path}")
            downloaded_paths.append(path)
            total_space += path.stat().st_size
        else:
            logger.warning(str(path_or_exception))

    logger.info(f"###\n"
                f"Downloaded {len(downloaded_paths)} feeds\n"
                f"Amounting to {total_space / (1024.0*1024.0)} MB")


if __name__ == "__main__":
    # Set start and end date for feeds
    start_date = datetime.date(2019, 1, 1)
    end_date = datetime.date(2019, 1, 8)  # datetime.date.today()

    # Start job
    download_and_store_gtfs(start_date, end_date, provider=TransitFeedProviders.GVB)
