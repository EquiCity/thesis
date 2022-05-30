from typing import Tuple, List
import igraph as ig
import pandas as pd
from ..rewards.egalitarian import egalitarian


def greedy_baseline(g: ig.Graph, census_data: pd.DataFrame, edge_types: List[str],
                    budget: int = 5, reward_func: callable = egalitarian) -> Tuple[List[float], List[ig.Edge]]:

    g_prime = g.copy()

    assert budget > 0

    removed_edges = []
    rewards_per_removal = []

    for i in range(budget):
        removable_edges = g_prime.es.select(type_in=edge_types)
        all_rewards = {}
        for edge in removable_edges:
            g_star = g_prime.copy()
            g_star.delete_edges(edge)
            r = reward_func(g_star, census_data)
            all_rewards[r] = edge
        max_reward = max(all_rewards.keys())
        edge_to_remove = all_rewards[max_reward]
        removed_edges.append(edge_to_remove)
        rewards_per_removal.append(max_reward)
        g_prime.delete_edges(edge_to_remove)

    return rewards_per_removal, removed_edges
