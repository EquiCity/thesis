import igraph as ig
from typing import Tuple, List, Optional
import abc

import numpy as np
import pandas as pd

import torch


class AbstractDeepQLearner(abc.ABC):
    def __init__(self, base_graph: ig.Graph, reward_function: callable, census_data: pd.DataFrame,
                 edge_types: List[str], budget: int, epochs: int, step_size: float = 0.1,
                 discount_factor: float = 1.0, learning_rate: float = 0.03,
                 loss_fn = torch.nn.MSELoss, optimizer = torch.optim.Adam) -> None:

        self.base_graph = base_graph
        self.reward_function = reward_function
        self.census_data = census_data
        self.epochs = epochs
        self.alpha = step_size
        self.gamma = discount_factor

        self.goal = budget

        self.actions = np.array([e.index for e in self.base_graph.es.select(type_in=edge_types)])
        self.starting_state: np.array = np.zeros(self.actions.size)

        self.wrong_action_reward: int = -100

        self.model = self.setup_model()
        self.loss_fn = loss_fn()
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

        self.trained = False

    @abc.abstractmethod
    def setup_model(self) -> torch.nn.Sequential:
        raise NotImplementedError()

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
    def choose_action(self, q_values, epsilon) -> int:
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(range(len(self.actions)))
        else:
            return np.random.choice([action_ for action_, value_ in enumerate(q_values) if value_ == np.max(q_values)])

    @abc.abstractmethod
    def train(self, return_rewards_over_epochs: bool = False) -> Optional[List[float]]:
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
