import os
import re
import datetime

import numpy as np
import tqdm
from gtfsmerger import GTFSMerger


DATA_PATH = os.environ.get("DATA_PATH", './data')


class TreeGTFSMerger:
    def __init__(self, fpaths, max_size, parent: bool = True):
        self.fpaths = fpaths
        self.max_size = max_size
        self.parent = parent

    def recursive_merge(self):
        while len(self.fpaths) > self.max_size:
            splits = np.array_split(self.fpaths, self.max_size)
            merged_paths = []

            if self.parent:
                for split in tqdm.tqdm(splits):
                    merged_paths.append(TreeGTFSMerger(split, self.max_size, parent=False).recursive_merge())
            else:
                for split in splits:
                    merged_paths.append(TreeGTFSMerger(split, self.max_size, parent=False).recursive_merge())

            self.fpaths = merged_paths

        if len(self.fpaths) > 1:
            dates_str = [re.findall(r'\d+', f) for f in self.fpaths]
            dates_str = [item for sublist in dates_str for item in sublist]
            date_converter = lambda x: x[0:4] + '-' + x[4:6] + '-' + x[6:9]
            dates = [datetime.date.fromisoformat(date_converter(d)) for d in dates_str]

            # Generate out path
            start_date = max(dates).isoformat().replace('-', '')
            end_date = min(dates).isoformat().replace('-', '')
            base_path = os.path.dirname(DATA_PATH)
            out_path = os.path.join(base_path, f"merged-gtfs-{start_date}-{end_date}.zip")

            if not os.path.exists(out_path):
                # Merge gtfs
                gtfs_merger = GTFSMerger()
                gtfs_merger.merge_from_fpaths(self.fpaths)
                # Store zip
                gtfs_merger.get_zipfile(out_path)
                del gtfs_merger

        elif len(self.fpaths) == 1:
            out_path = self.fpaths[0]

        return out_path
