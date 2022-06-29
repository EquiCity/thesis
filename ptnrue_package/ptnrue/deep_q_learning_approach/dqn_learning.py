from typing import Optional, List

import numpy as np
import torch
from .abstract_deep_q_learner import AbstractDeepQLearner
import logging
from tqdm import tqdm


logging.basicConfig()
logger = logging.getLogger('graph_extraction')
logger.setLevel(logging.INFO)


class DeepQLearner(AbstractDeepQLearner):

    def setup_model(self) -> torch.nn.Sequential:
        # Q-function layer definition
        l1 = len(self.starting_state)  # states
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

    def train(self, return_rewards_over_episodes: bool = True, verbose: bool = True) -> Optional[List[float]]:
        rewards_over_episodes = []
        epsilon = 1.0

        episode_iterator = tqdm(range(self.episodes)) if verbose else range(self.episodes)

        for ep in episode_iterator:
            state = self.starting_state

            while state[state == 1].size()[0] != self.goal:
                # Choose action with epsilon-greedy strategy
                action_ = self.choose_action(state, epsilon)
                next_state, reward = self.step(state, action_)

                # self.replay_buffer.add((state, action_, reward, next_state))

                with torch.no_grad():
                    newQ = self.model(next_state.float())

                maxQ = torch.max(newQ)

                if next_state[next_state == 1].size()[0] == self.goal:
                    Y = reward
                else:
                    Y = reward + self.gamma * maxQ

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
