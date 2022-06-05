from typing import Tuple, List
import igraph as ig
import pandas as pd
from ..rewards.egalitarian import egalitarian_theil


def greedy_baseline(g: ig.Graph, census_data: pd.DataFrame, edge_types: List[str],
                    budget: int = 5, reward_func: callable = egalitarian_theil) -> Tuple[List[float], List[ig.Edge]]:

    assert budget > 0

    removable_edges = g.es.select(type_in=edge_types)

    removed_edges = []
    rewards_per_removal = []

    for i in range(budget):
        all_rewards = {}
        for edge in removable_edges:
            g_prime = g.copy()
            g_prime.delete_edges(removed_edges + [edge])
            r = reward_func(g_prime, census_data)
            all_rewards[r] = edge

        max_reward = max(all_rewards.keys())
        edge_to_remove = all_rewards[max_reward]
        removed_edges.append(edge_to_remove)
        rewards_per_removal.append(max_reward)

    return rewards_per_removal, removed_edges
