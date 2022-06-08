import igraph as ig
import pandas as pd
from typing import List
from experiments.constants.travel_metric import TravelMetric


def simple(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None,
              metrics: List[TravelMetric] = None, com_threshold: float = 12) -> float:
    """

    Args:
        com_threshold:
        groups:
        g:
        census_data:
        group:

    Returns:

    """

    return sum(g.es.select(type_in=['train','bus'],active_eq=1)['distance'])
