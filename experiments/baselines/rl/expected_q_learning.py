from .abstract_q_learner import AbstractQLearner
import numpy as np
from typing import List
from tqdm import tqdm


class ExpectedQLearner(AbstractQLearner):

    def train(self, return_rewards_over_epochs: bool = True, verbose: bool = True) -> List[float]:
        if self.trained:
            raise RuntimeError("Cannot run training pipeline twice. Please create a new learner object")

        rewards_over_epochs = []
        iterator = tqdm(range(self.epochs)) if verbose else range(self.epochs)
        for i in iterator:
            ord_state = self.get_state_key(self.starting_state)
            rewards = 0
            while len(ord_state) != self.goal:
                action = self.choose_action(ord_state, epsilon=1 / (i + 1))
                next_state, reward = self.step(ord_state, action)
                next_ord_state = self.get_state_key(next_state)
                rewards += reward
                # Q-Learning update
                self.q_values[ord_state][action] += self.alpha * (
                                                    reward + self.gamma * np.max(self.q_values[next_ord_state]) -
                                                    self.q_values[ord_state][action])
                ord_state = next_ord_state

        self.trained = True

        if return_rewards_over_epochs:
            return rewards_over_epochs
