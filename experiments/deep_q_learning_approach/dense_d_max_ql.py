from typing import Optional, List

import numpy as np
import torch
import random
from matplotlib import pylab as plt
from .abstract_deep_q_learner import AbstractDeepQLearner
import igraph as ig
import pandas as pd
import logging


logging.basicConfig()
logger = logging.getLogger('graph_extraction')
logger.setLevel(logging.INFO)


class DeepMaxQLearner(AbstractDeepQLearner):

    def setup_model(self) -> torch.nn.Sequential:
        # Q-function layer definition
        l1 = len(self.actions)  # states
        l2 = 150
        l3 = 100
        l4 = len(self.actions)  # actions

        return torch.nn.Sequential(
            torch.nn.Linear(l1, l2),
            torch.nn.ReLU(),
            torch.nn.Linear(l2, l3),
            torch.nn.ReLU(),
            torch.nn.Linear(l3, l4)
        )

    def train(self, return_rewards_over_epochs: bool = True, verbose: bool = True) -> Optional[List[float]]:
        rewards_over_epochs = []

        for i in range(self.epochs):
            state_ = np.zeros(len(self.actions))
            state = torch.from_numpy(state_).float()
            epsilon = 1.0

            max_reward = -np.inf

            while state[state == 1].size != self.goal:
                self.q_values = self.model(state)
                action_ = self.choose_action(state, epsilon)
                next_state_, reward = self.step(state, action_)

                next_state = torch.from_numpy(next_state_).float()
                reward = self.reward_function(next_state)

                with torch.no_grad():
                    newQ = self.model(next_state)

                maxQ = torch.max(newQ)

                if reward == self.wrong_action_reward:
                    Y = reward + (self.gamma * maxQ)
                else:
                    Y = reward

                Y = torch.Tensor([Y]).detach()
                X = self.q_values.squeeze()[action_]
                loss = self.loss_fn(X, Y)

                self.optimizer.zero_grad()
                loss.backward()
                rewards_over_epochs.append(loss.item())
                self.optimizer.step()
                state = next_state

            if epsilon > 0.01:
                epsilon -= (1 / self.epochs)

        if return_rewards_over_epochs:
            return rewards_over_epochs
