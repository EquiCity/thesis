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

    def train(self, return_rewards_over_episodes: bool = True, verbose: bool = True) -> Optional[List[float]]:
        rewards_over_episodes = []
        episodic_loss = []
        epsilon = 1.0
        step_counter = 0

        episode_iterator = tqdm(range(self.episodes)) if verbose else range(self.episodes)

        # for each episode
        for i_episode, ep in enumerate(episode_iterator):
            state = self.starting_state

            # while the state is not the terminal state
            while state[state == 1].size()[0] != self.goal:

                # Choose action with epsilon-greedy strategy
                action_ = self.choose_action(state, epsilon)
                next_state, reward = self.step(state, action_)

                # Store this transistion as an experience in the replay buffer
                if len(self.memory) < self.replay_memory_size:
                    self.memory.push(state, action_, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            rewards_over_episodes.append(reward)


        self.trained = True

        if return_rewards_over_episodes:
            return rewards_over_episodes
