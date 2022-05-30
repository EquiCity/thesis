from .abstract_q_learner import AbstractQLearner
import numpy as np


class MaxQLearner(AbstractQLearner):

    def train(self) -> None:
        if self.trained:
            raise RuntimeError("Cannot run training pipeline twice. Please create a new learner object")

        for i in range(self.epochs):
            ord_state = self.get_state_key(self.starting_state)
            max_reward = -np.inf
            while len(ord_state) != self.goal:
                action = self.choose_action(ord_state, epsilon=1 / (i + 1))
                next_state, reward = self.step(ord_state, action)
                next_ord_state = self.get_state_key(next_state)
                # TODO: check if this is correct
                max_reward = np.max([max_reward, reward])
                # Q-Learning update
                self.q_values[ord_state][action] += (1 - self.alpha) * self.q_values[next_ord_state] + \
                                                    self.alpha * (reward + self.gamma *
                                                                  np.max(self.q_values[next_ord_state]))
                ord_state = next_ord_state

        self.trained = True
