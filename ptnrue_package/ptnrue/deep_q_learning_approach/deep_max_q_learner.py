import math
from typing import Optional, List, Tuple

import numpy as np
import torch
from .abstract_deep_q_learner import AbstractDeepQLearner
import logging
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeepMaxQLearner(AbstractDeepQLearner):

    def _compute_q_learning_targets(self, reward_batch: torch.tensor, next_state_values: torch.tensor) -> torch.tensor:
        return torch.max(reward_batch, (self.gamma * next_state_values))

    def train(self, return_rewards_over_episodes: bool = True,
              return_eps_values: bool = True, verbose: bool = True) -> Optional[Tuple[List[float], List[float]]]:
        max_rewards_over_episodes = []
        eps_values = []

        episode_iterator = tqdm(range(self.episodes)) if verbose else range(self.episodes)

        # for each episode
        for i_episode, ep in enumerate(episode_iterator):
            state = self.starting_state
            max_reward = -math.inf

            # while the state is not the terminal state
            while state[state == 1].size()[0] < self.goal:

                # Choose action with epsilon-greedy strategy
                epsilon = self.eps_schedule.get_current_eps()
                action_ = self.choose_action(state, epsilon)
                next_state, reward = self.step(state, action_)

                max_reward = max([reward.item(), max_reward])

                # Store this transitions as an experience in the replay buffer
                # if len(self.memory) < self.replay_memory_size:
                available_actions_next_state = self._get_available_actions(next_state)
                available_actions_next_state_t = self._get_available_actions_boolean_tensor(available_actions_next_state)
                # 'state', 'action', 'next_state', 'available_actions_next_state', 'reward'
                self.memory.push(state, action_, next_state, available_actions_next_state_t, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            eps_values.append(self.eps_schedule.get_current_eps())
            max_rewards_over_episodes.append(max_reward)

        self.trained = True

        output = []
        if return_rewards_over_episodes:
            output.append(max_rewards_over_episodes)

        if return_eps_values:
            output.append(eps_values)

        return tuple(output)

    def inference(self) -> Tuple[List[float], List[int]]:
        rewards_per_removal, final_state = super(DeepMaxQLearner, self).inference()

        # Crop until max
        max_idx = np.argmax(rewards_per_removal) + 1

        return rewards_per_removal[:max_idx], final_state[:max_idx]