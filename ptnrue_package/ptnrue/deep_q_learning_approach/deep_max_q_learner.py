from typing import Optional, List

import numpy as np
import torch
from .abstract_deep_q_learner import AbstractDeepQLearner
import logging
from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger('graph_extraction')
logger.setLevel(logging.INFO)


class DeepMaxQLearner(AbstractDeepQLearner):

    def train(self, return_rewards_over_episodes: bool = True, verbose: bool = True) -> Optional[List[float]]:
        rewards_over_episodes = []
        epsilon = 1.0

        iterator = tqdm(range(self.episodes)) if verbose else range(self.episodes)

        for i in iterator:
            state = self.starting_state

            max_reward = -np.inf

            while state[state == 1].size()[0] != self.goal:
                self.q_values = self.model(state.float())
                action_ = self.choose_action(state, epsilon)
                next_state, reward = self.step(state, action_)

                with torch.no_grad():
                    newQ = self.model(next_state.float())

                maxQ = torch.max(newQ)

                if reward == self.wrong_action_reward:
                    Y = reward + (self.gamma * maxQ)
                else:
                    Y = reward

                Y = torch.tensor(Y, dtype=torch.float).detach()
                X = self.q_values.squeeze()[action_]
                loss = self.loss_fn(X, Y)

                self.optimizer.zero_grad()
                loss.backward()
                rewards_over_episodes.append(loss.item())
                self.optimizer.step()
                state = next_state

            rewards_over_episodes.append(reward)

            if epsilon > 0.1:
                epsilon -= 0.01

        self.trained = True

        if return_rewards_over_episodes:
            return rewards_over_episodes
