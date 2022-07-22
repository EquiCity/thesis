import datetime
import pickle

import torch
import random
from eptnr.deep_q_learning_approach.deep_q_learner import DeepQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from eptnr.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)
from eptnr.plotting.deep_solution_plotting import plot_nn_loss_reward_epsilon
import logging
from eptnr.plotting.solution_plotting import plot_rewards_and_graphs
from eptnr.plotting.policy_plotting import PolicyPlotter
from matplotlib import pyplot as plt
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 2
    g = ig.load(Path(f"../base_data/graph_{dataset}.gml"))
    census_data = gpd.read_file(Path(f"../base_data/census_data_{dataset}.geojson"))
    # reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 3
    com_threshold = 15
    # reward = CustomReward(reward_dict=reward_dict, census_data=census_data,
    #                       com_threshold=com_threshold)
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=com_threshold)

    episodes = 1000

    # Replay Memory
    batch_size = 64
    replay_memory_size = 512
    target_network_update_step = 100

    # EPS Schedule
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 50
    static_eps_steps = 500 * budget

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    q_learner = DeepQLearner(base_graph=g, reward=reward, budget=budget, edge_types=edge_types,
                             target_network_update_step=target_network_update_step,
                             episodes=episodes, batch_size=batch_size,
                             replay_memory_size=replay_memory_size,
                             eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                             static_eps_steps=static_eps_steps)

    cum_rewards_over_episodes, max_rewards_over_episodes, eps_values_over_episodes = q_learner.train()
    rewards, edges = q_learner.inference()

    sub_sampled_policy_net_loss = q_learner.policy_net_loss[0::budget]
    sub_sampled_policy_net_loss = np.concatenate([np.repeat(sub_sampled_policy_net_loss[0], batch_size//budget),
                                                 sub_sampled_policy_net_loss])
    fig, ax = plot_nn_loss_reward_epsilon(sub_sampled_policy_net_loss,
                                          {
                                              'maximum reward': max_rewards_over_episodes,
                                              'cumulative reward': cum_rewards_over_episodes,
                                          },
                                          eps_values_over_episodes,
                                          title='Deep Q-learning network loss\nand policy performance')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'figures/synth_ds_{dataset}_deep_q_learning_loss_and_reward.png')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_deep_q_learning_policy.png')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'figures/synth_ds_{dataset}_deep_q_learning_loss_and_reward.svg')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_deep_q_learning_policy.svg')
    plt.show()

    # Plot the policy
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    title = f""
    _, _ = PolicyPlotter().from_model(model=q_learner.policy_net, budget=budget,
                                      actions=q_learner.actions.tolist(),
                                      title=title,
                                      fig=fig, ax=ax)
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'figures/synth_ds_{dataset}_deep_q_learning_policy.png')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_deep_q_learning_policy.png')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'figures/synth_ds_{dataset}_deep_q_learning_policy.svg')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_deep_q_learning_policy.svg')

    plt.show()
    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()
    # logger.info(f"Removed edges: {edges}")
    # q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
