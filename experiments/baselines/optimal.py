import itertools as it
from typing import Tuple, List
import igraph as ig
import numpy as np
import pandas as pd
from experiments.rewards.egalitarian import egalitarian_theil
from tqdm import tqdm
from experiments.constants.travel_metric import TravelMetric


def optimal_baseline(g: ig.Graph, census_data: pd.DataFrame, edge_types: List[str],
                     budget: int = 5, reward_func: callable = egalitarian_theil,
                     groups: List[str] = None, metrics: List[TravelMetric] = None,
                     com_threshold: float = 15) -> List[Tuple[List[float], List[int]]]:
    if groups is None:
        groups = []
    assert budget > 0

    removable_edges = g.es.select(type_in=edge_types, active_eq=1)
    possible_combinations = [[e.index for e in es] for es in it.combinations(removable_edges, budget)]

    rewards = - np.ones(len(possible_combinations)) * np.inf

    for i, candidate in enumerate(tqdm(possible_combinations)):
        g_prime = g.copy()
        g_prime.es[candidate]['active'] = 0
        rewards[i] = reward_func(g=g_prime, census_data=census_data, groups=groups,
                                 metrics=metrics, com_threshold=com_threshold)

    max_reward_candidates_idxs = np.where(rewards == rewards.max())[0]

    optimal_solutions_and_rewards_per_removal = []

    for cand_i in max_reward_candidates_idxs:
        es_idx_list = possible_combinations[cand_i]
        rewards_per_removal = []
        for i in range(0, budget):
            g_prime = g.copy()
            g_prime.es[es_idx_list[0:i+1]]['active'] = 0
            rewards_per_removal.append(reward_func(g_prime, census_data, groups, metrics))

        optimal_solutions_and_rewards_per_removal.append((rewards_per_removal, es_idx_list))

    return optimal_solutions_and_rewards_per_removal
