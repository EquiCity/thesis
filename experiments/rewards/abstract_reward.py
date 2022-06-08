import abc
import igraph as ig
import pandas as pd
from typing import List, Dict
from experiments.constants.travel_metric import TravelMetric
from _utils import get_tt_hops_com_dfs


class AbstractReward(abc.ABC):

    def __init__(self, census_data: pd.GeoDataFrame, metrics: List[TravelMetric] = None,
                 groups: List[str] = None, com_threshold: float = 12) -> None:

        self.census_data = census_data
        self.com_threshold = com_threshold

        self.metrics = metrics if metrics is not None else [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]

        self.metrics_names = [t.value for t in metrics]

        self.groups = groups if groups else None

    def retrieve_dfs(self, g: ig.Graph) -> Dict[pd.DataFrame]:
        g_prime = g.subgraph_edges(g.es.select(active_eq=1), delete_vertices=False)
        tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g_prime, self.census_data, self.com_threshold)

        metrics_values = {
            TravelMetric.TT.value: tt_samples,
            TravelMetric.HOPS.value: hops_samples,
            TravelMetric.COM.value: com_samples
        }

        self.groups = list(tt_samples.group.unique()) if not self.groups else self.groups
        metrics_dfs = {metrics_name: metrics_values[metrics_name] for metrics_name in self.metrics_names}

        return metrics_dfs

    @abc.abstractmethod
    def evaluate(self, g: ig.Graph) -> float:
        raise NotImplementedError()
