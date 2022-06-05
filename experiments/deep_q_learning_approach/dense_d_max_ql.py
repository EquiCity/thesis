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

    def train(self, return_rewards_over_epochs: bool = False) -> Optional[List[float]]:
        rewards_over_epochs = []

        for i in range(self.epochs):
            state_ = np.zeros(len(self.actions))
            state = torch.from_numpy(state_).float()
            epsilon = 1.0

            max_reward = -np.inf

            while state[state == 1].size != self.goal:
                qval = self.model(state)
                action_ = self.choose_action(state, epsilon)

                action = self.actions[action_]
                next_state_, reward = self.step(state, action)
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
                X = qval.squeeze()[action_]
                loss = self.loss_fn(X, Y)
                logger.info(i, loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                rewards_over_epochs.append(loss.item())
                self.optimizer.step()
                state = next_state

            if epsilon > 0.01:
                epsilon -= (1 / self.epochs)

        plt.figure(figsize=(10, 7))
        plt.plot(rewards_over_epochs)
        plt.xlabel("Epochs", fontsize=22)
        plt.ylabel("Loss", fontsize=22)

        if return_rewards_over_epochs:
            return rewards_over_epochs
