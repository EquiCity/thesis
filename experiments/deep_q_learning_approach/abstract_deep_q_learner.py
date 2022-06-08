import igraph as ig
from typing import Tuple, List, Optional
import abc

import numpy as np
import pandas as pd

import torch
from experiments.baselines.rl.abstract_q_learner import AbstractQLearner


class AbstractDeepQLearner(AbstractQLearner, abc.ABC):
    def __init__(self, base_graph: ig.Graph, reward_function: callable, census_data: pd.DataFrame,
                 edge_types: List[str], budget: int, epochs: int, step_size: float = 0.1, discount_factor: float = 1.0,
                 learning_rate: float = 0.03, loss_fn=torch.nn.MSELoss, optimizer=torch.optim.Adam) -> None:
        super().__init__(base_graph, reward_function, census_data, edge_types, budget, epochs, step_size,
                         discount_factor)

        self.starting_state: np.array = np.zeros(self.actions.size)
        self.wrong_action_reward: int = -100

        self.actions = np.array([e.index for e in self.base_graph.es.select(type_in=edge_types, active_eq=1)])
        self.q_values = None

        self.model = self.setup_model()
        self.loss_fn = loss_fn()
        self.learning_rate = learning_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.learning_rate)

    @abc.abstractmethod
    def setup_model(self) -> torch.nn.Sequential:
        raise NotImplementedError()

    def step(self, state: Tuple[int], action_idx: int) -> Tuple[np.array, float]:
        # TODO: Consider scaling the probabilities of not-allowed actions
        g_prime = self.base_graph.copy()

        try:
            edge_idx = self.actions[action_idx]
            if edge_idx in state:
                raise ValueError("Cannot choose same action twice")
            next_state = state + (edge_idx,)
            g_prime.es[list(next_state)]['active'] = 0
            reward = self.reward_function(g_prime, self.census_data)
        except ValueError:
            reward = self.wrong_action_reward
            next_state = self.starting_state

        return next_state, reward

    # choose an action based on epsilon greedy algorithm
    def choose_action(self, state: np.array, epsilon: float) -> int:
        available_actions = [action_idx for action_idx, action in enumerate(self.actions)
                             if action not in list(state)]

        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(available_actions)
        else:
            values_ = self.q_values[state]
            choosable_actions = [action_ for action_, value_ in enumerate(values_)
                                 if value_ == np.max(values_[available_actions]) and action_ in available_actions]
            return np.random.choice(choosable_actions)

    @abc.abstractmethod
    def train(self, return_rewards_over_epochs: bool = True, verbose: bool = True) -> Optional[List[float]]:
        raise NotImplementedError()

    def inference(self) -> Tuple[List[float], List[int]]:
        if not self.trained:
            raise RuntimeError("Please run the training before inference")

        state = self.starting_state
        rewards_per_removal = []

        for i in range(self.goal):
            action_idx = self.choose_action(state, 0)
            state, reward = self.step(state, action_idx)
            rewards_per_removal.append(reward)

        # A state is nothing else than an indicator as of whether an edge
        # is removed or not, i.e. whether an action was enacted or not.
        # Hence, we use the state to take out the actions, i.e. edge indexes
        # which represent the final state
        final_state = self.actions[state.astype(np.bool)]
        return rewards_per_removal, final_state
