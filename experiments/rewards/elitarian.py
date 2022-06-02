import igraph as ig
import pandas as pd
# from .egalitarian import egalitarian
from .utilitarian import utilitarian
from typing import List
from experiments.constants.travel_metric import TravelMetric


def elitarian(g: ig.Graph, census_data: pd.DataFrame, groups: List[str] = None,
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

    return utilitarian(g, census_data, groups, com_threshold)  # + egalitarian(g, census_data, [group])
