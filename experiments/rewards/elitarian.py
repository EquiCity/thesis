import igraph as ig
import pandas as pd
# from .egalitarian import egalitarian
from .utilitarian import utilitarian


def elitarian(g: ig.Graph, census_data: pd.DataFrame, group: str) -> float:
    """

    Args:
        g:
        census_data:
        group:

    Returns:

    """

    return utilitarian(g, census_data, [group])  # + egalitarian(g, census_data, [group])
