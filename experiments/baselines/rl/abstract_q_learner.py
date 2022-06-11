import os
import pickle

import igraph as ig
from typing import Tuple, List, Optional, Union
import abc

from experiments.rewards import BaseReward

import numpy as np
import pandas as pd

import itertools as it
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class AbstractQLearner(abc.ABC):
    def __init__(self, base_graph: ig.Graph, reward: BaseReward,
                 edge_types: List[str], budget: int, episodes: int, step_size: float = 0.1,
                 discount_factor: float = 1.0) -> None:
        self.base_graph = base_graph
        self.reward = reward
        self.episodes = episodes
        self.alpha = step_size
        self.gamma = discount_factor

        self.goal = budget
        self.starting_state: Tuple[int] = ()
        self.wrong_action_reward: int = -100

        self.actions = np.array([e.index for e in self.base_graph.es.select(type_in=edge_types, active_eq=1)])

        if len(self.actions) <= self.goal:
            raise ValueError(f"Can only choose {len(self.actions)} edges, "
                             f"hence max budget is {len(self.actions) - 1}. "
                             f"Budget {self.goal} not possible.")

        self.q_values = {
            self.get_state_key(tuple(e)): np.zeros(len(self.actions), dtype=np.float)
            for k in range(self.goal + 1) for e in it.combinations(self.actions, k)
        }

        self.state_visit = [{key: 1 for key in self.q_values.keys()} for _ in range(self.goal+1)]

        self.trained = False

    @staticmethod
    def get_state_key(removed_edges: Tuple) -> Tuple:
        return tuple(np.sort(list(removed_edges)))

    def step(self, state: Tuple[int], action_idx: int) -> Tuple[Tuple[int], float]:
        # TODO: Consider scaling the probabilities of not-allowed actions
        g_prime = self.base_graph.copy()

        try:
            edge_idx = self.actions[action_idx]
            if edge_idx in state:
                raise ValueError("Cannot choose same action twice")
            next_state = state + (edge_idx,)
            g_prime.es[list(next_state)]['active'] = 0
            reward = self.reward.evaluate(g_prime)
            # if reward > 50:
            #     logger.info(f"BIG REWARD {reward}")
            optimal = {14}
            if set(next_state).issubset(optimal):
                if set(next_state) == optimal:
                    # logger.info("REACHED OPTIMUM")
                    reward += 1000
            # reward += 100
            self.state_visit[len(next_state)][self.get_state_key(next_state)] += 1

        except ValueError:
            reward = self.wrong_action_reward
            next_state = self.starting_state

        return next_state, reward

    # choose an action based on epsilon greedy algorithm
    def choose_action(self, state: tuple, epsilon: float) -> int:
        available_actions = [action_idx for action_idx, action in enumerate(self.actions)
                             if action not in list(state)]

        assert len(available_actions) > 0

        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(available_actions)
        else:
            values_ = self.q_values[state]
            choosable_actions = [action_ for action_, value_ in enumerate(values_)
                                 if value_ == np.max(values_[available_actions]) and action_ in available_actions]
            return np.random.choice(choosable_actions)

    @abc.abstractmethod
    def train(self, return_rewards_over_episodes: bool = True, verbose: bool = True) -> Optional[List[float]]:
        raise NotImplementedError()

    def save_model(self, fpath: Union[str, os.PathLike]):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, fpath: Union[str, os.PathLike]):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    def inference(self) -> Tuple[List[float], List[int]]:
        if not self.trained:
            raise RuntimeError("Please run the training before inference")

        ord_state = self.get_state_key(self.starting_state)
        rewards_per_removal = []

        for i in range(self.goal):
            action_idx = self.choose_action(ord_state, 0)
            next_state, reward = self.step(ord_state, action_idx)
            ord_state = self.get_state_key(next_state)
            rewards_per_removal.append(reward)

        final_state = list(ord_state)
        return rewards_per_removal, final_state
