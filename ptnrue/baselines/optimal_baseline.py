import itertools as it
from typing import Tuple, List
import igraph as ig
import numpy as np
from ..rewards import BaseReward
from tqdm import tqdm
import logging
from .utils.compute_rewards import compute_rewards_over_removals

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def optimal_baseline(g: ig.Graph, reward: BaseReward, edge_types: List[str],
                     budget: int = 5) -> List[Tuple[List[float], List[int]]]:

    assert budget > 0
    assert budget < len(g.es.select(type_in=edge_types))

    removable_edges = g.es.select(type_in=edge_types, active_eq=1)
    possible_combinations = [[e.index for e in es] for es in it.combinations(removable_edges, budget)]
    logger.info(f"Possible states: {possible_combinations}")
    rewards = -np.ones(len(possible_combinations)) * np.inf

    for i, candidate in enumerate(tqdm(possible_combinations)):
        g_prime = g.copy()
        g_prime.es[candidate]['active'] = 0
        rewards[i] = reward.evaluate(g_prime)
        logger.info(f"For state {candidate} obtained rewards {rewards[i]}")

    max_reward_candidates_idxs = np.where(rewards == rewards.max())[0]

    optimal_solutions_and_rewards_per_removal = []
    logger.info("OPTIMAL STATES:")
    for cand_i in max_reward_candidates_idxs:
        logger.info(f"For state {possible_combinations[cand_i]} obtained rewards {rewards[cand_i]}")
        es_idx_list = possible_combinations[cand_i]
        rewards_per_removal = compute_rewards_over_removals(g, budget, reward, es_idx_list)
        optimal_solutions_and_rewards_per_removal.append((rewards_per_removal, es_idx_list))

    return optimal_solutions_and_rewards_per_removal
