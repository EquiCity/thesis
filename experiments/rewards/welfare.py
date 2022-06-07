import igraph as ig
import pandas as pd
from typing import List
from ._utils import get_tt_hops_com_dfs
from experiments.constants.travel_metric import TravelMetric


def welfare(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None,
            metrics: List[TravelMetric] = None, com_threshold: float = 12) -> float:
    """
    This reward gives most weighting to improve the
    Args:
        g:
        census_data:
        groups:

    Returns:

    """
    g_prime = g.subgraph_edges(g.es.select(active_eq=1), delete_vertices=False)
    tt_samples, hops_samples, com_samples = get_tt_hops_com_dfs(g_prime, census_data, com_threshold)
    raise NotImplementedError()
