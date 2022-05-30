from .abstract_q_learner import AbstractQLearner
import numpy as np


class SARSALearner(AbstractQLearner):

    def train(self, expected: bool = False) -> None:
        if self.trained:
            raise RuntimeError("Cannot run training pipeline twice. Please create a new learner object")

        for i in range(self.epochs):
            ord_state = self.get_state_key(self.starting_state)
            action = self.choose_action(ord_state, self.q_values)
            rewards = 0.0
            epsilon = 1/(i+1)
            while len(ord_state) != self.goal:
                next_state, reward = self.step(ord_state, action)
                next_ord_state = self.get_state_key(next_state)
                next_action = self.choose_action(next_ord_state, self.q_values)
                rewards += reward
                if not expected:
                    target = self.q_values[next_ord_state][next_action]
                else:
                    # calculate the expected value of new state
                    target = 0.0
                    q_next = self.q_values[next_ord_state]
                    best_actions = np.argwhere(q_next == np.max(q_next))
                    for action_ in self.actions:
                        if action_ in best_actions:
                            target += ((1.0 - epsilon) / len(best_actions) + epsilon / len(self.actions)) \
                                      * self.q_values[next_ord_state][action_]
                        else:
                            target += epsilon / len(self.actions) * self.q_values[next_ord_state][action_]
                target *= self.gamma
                self.q_values[ord_state][action] += self.alpha * (
                        reward + target - self.q_values[ord_state][action])
                # Updating
                ord_state = next_ord_state
                action = next_action

        self.trained = True
