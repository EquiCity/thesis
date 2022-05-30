import igraph as ig
from typing import Tuple, List
import abc

import numpy as np
import pandas as pd

import itertools as it


class AbstractQLearner(abc.ABC):
    def __init__(self, base_graph: ig.Graph, reward_function: callable, census_data: pd.DataFrame,
                 edge_types: List[str], budget: int, epochs: int, step_size: float = 0.1,
                 discount_factor: float = 1.0) -> None:
        self.base_graph = base_graph
        self.reward_function = reward_function
        self.census_data = census_data
        self.epochs = epochs
        self.alpha = step_size
        self.gamma = discount_factor

        self.goal = budget
        self.starting_state: Tuple[int] = ()
        self.wrong_action_reward: int = -100

        self.actions = [e.index for e in self.base_graph.es.select(type_in=edge_types)]

        self.q_values = {
            self.get_state_key(tuple(e)): np.array([0 if i not in e else -100 for i in range(len(self.actions))])
            for k in range(self.goal+1) for e in it.combinations(self.actions, k)
        }

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
            g_prime.delete_edges(
                [(e.source_vertex.index, e.target_vertex.index) for e in self.base_graph.es[list(next_state)]])
            reward = self.reward_function(g_prime, self.census_data)
        except ValueError:
            reward = self.wrong_action_reward
            next_state = self.starting_state

        return next_state, reward

    # choose an action based on epsilon greedy algorithm
    def choose_action(self, state: tuple, epsilon: float) -> int:
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(range(len(self.actions)))
        else:
            values_ = self.q_values[state]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    @abc.abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    def inference(self) -> Tuple[float, List[ig.Edge]]:
        if not self.trained:
            raise RuntimeError("Please run the training before inference")

        ord_state = self.get_state_key(self.starting_state)
        final_reward = -np.inf

        for i in range(self.goal):
            action_idx = self.choose_action(ord_state, 1/(i+1))
            next_state, reward = self.step(ord_state, action_idx)
            ord_state = self.get_state_key(next_state)
            final_reward = reward

        return final_reward, self.base_graph.es[ord_state]
