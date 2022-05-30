from typing import List, Tuple
import random
import igraph as ig
import pandas as pd
from ..rewards.egalitarian import egalitarian


def random_baseline(g: ig.Graph, census_data: pd.DataFrame, edge_types: List[str],
                    budget: int = 5, reward_func: callable = egalitarian) -> Tuple[List[float], List[ig.Edge]]:
    """

    Args:
        g:
        census_data:
        budget:
        reward_func:

    Returns:

    """
    g_prime = g.copy()

    assert budget > 0

    removed_edges = []
    rewards_per_removal = []

    for i in range(budget):
        removable_edges = g_prime.es.select(type_in=edge_types)
        edge_to_remove = random.sample(list(removable_edges), 1)[0]
        removed_edges.append(edge_to_remove)

        g_prime.delete_edges(edge_to_remove)
        r = reward_func(g_prime, census_data)
        rewards_per_removal.append(r)

    return rewards_per_removal, removed_edges
