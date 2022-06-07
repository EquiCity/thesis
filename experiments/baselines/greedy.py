from typing import Tuple, List
import igraph as ig
import pandas as pd
from ..rewards.egalitarian import egalitarian_theil


def greedy_baseline(g: ig.Graph, census_data: pd.DataFrame, edge_types: List[str],
                    budget: int = 5, reward_func: callable = egalitarian_theil) -> Tuple[List[float], List[int]]:

    assert budget > 0

    removed_edges = []
    rewards_per_removal = []

    g_prime = g.copy()

    for i in range(budget):
        removable_edges = g_prime.es.select(type_in=edge_types, active_eq=1)

        all_rewards = {}
        for edge in removable_edges:
            g_prime.es[edge.index]['active'] = 0
            r = reward_func(g_prime, census_data)
            g_prime.es[edge.index]['active'] = 1
            all_rewards[r] = edge.index

        max_reward = max(all_rewards.keys())
        edge_to_remove = all_rewards[max_reward]
        removed_edges.append(edge_to_remove)
        g_prime.es[edge_to_remove]['active'] = 0
        rewards_per_removal.append(max_reward)

    return rewards_per_removal, removed_edges
