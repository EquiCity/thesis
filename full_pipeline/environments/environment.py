import igraph as ig
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod
import pandas as pd


class AbstractEnvironment(ABC):

    @abstractmethod
    @property
    def G(self) -> ig.Graph:
        pass

    @abstractmethod
    @property
    def pois(self) -> pd.DataFrame:
        pass

    @abstractmethod
    @property
    def origins(self) -> pd.DataFrame:
        pass

    @abstractmethod
    @property
    def census_data(self) -> pd.DataFrame:
        pass


class GTFSEnvironment(AbstractEnvironment):

    def __init__(self, gtfs_path: Path, osmn_graph: Path, poi_paths: List[Path], origin_paths: List[Path]) -> None:
        self.gtfs_path = self._check_integrity_gtfs(gtfs_path)
        self.osmn_path = self._check_integrity_osm(osmn_graph)
        self.poi_paths = self._check_integrity_point_paths(poi_paths)
        self.origin_paths = self._check_integrity_point_paths(origin_paths)

    @classmethod
    def _check_integrity_gtfs(cls, path) -> Path:
        return path

    @classmethod
    def _check_integrity_osm(cls, path) -> Path:
        return path

    @classmethod
    def _check_integrity_point_paths(cls, paths: List[Path]) -> List[Path]:
        for path in paths:
            if path.suffix != '.csv':
                raise IOError(f"expected a '.csv' file rather than {path}")
            cols = open(path).readline().replace('\n', '').split(',')
            if not cols == ['id', 'lat', 'lon']:
                raise ValueError(f"expected file to have columns: id, lat, lon. Not: {cols}")

        return paths

    def _generate_gtfs_network(self):
        pass

    def _generate_osm_network(self):
        pass


