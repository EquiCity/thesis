import abc
import igraph as ig
import pandas as pd
from typing import List
from experiments.constants.travel_metric import TravelMetric
from _utils import get_tt_hops_com_dfs


class AbstractReward(abc.ABC):

    def __init__(self, g: ig.Graph, census_data: pd.GeoDataFrame, metrics: List[TravelMetric] = None,
                 groups: List[str] = None, com_threshold: float = 12) -> None:

        self.g = g
        self.census_data = census_data
        self.com_threshold = com_threshold

        self.tt_samples, self.hops_samples, self.com_samples = get_tt_hops_com_dfs(g, census_data, com_threshold)

        self.metrics_values = {
            TravelMetric.TT.value: self.tt_samples,
            TravelMetric.HOPS.value: self.hops_samples,
            TravelMetric.COM.value: self.com_samples
        }

        self.metrics = metrics if metrics is not None else [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]

        self.metrics_names = [t.value for t in metrics]
        self.metrics_dfs = {metrics_name: self.metrics_values[metrics_name] for metrics_name in self.metrics_names}

        self.groups = list(self.tt_samples.group.unique()) if not groups else groups

    @abc.abstractmethod
    def compute(self) -> float:
        raise NotImplementedError()
