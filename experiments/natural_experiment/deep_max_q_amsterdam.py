import datetime
import os
import pickle

from ptnrue.deep_q_learning_approach.deep_max_q_learner import DeepMaxQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)
import torch
import random
import logging
from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
from matplotlib import pyplot as plt
from pathlib import Path
import ray

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

ray.init(ignore_reinit_error=True)

GRAPH_PATH = Path(os.environ['GRAPH_PATH'])
CENSUS_PARQUET_PATH = Path(os.environ['CENSUS_PARQUET_PATH'])


if __name__ == "__main__":
    g: ig.Graph = ig.load(GRAPH_PATH)
    census_data = gpd.read_parquet(CENSUS_PARQUET_PATH)

    census_data['n_inh'] = census_data['a_inw']
    census_data['neighborhood'] = census_data['BU_NAAM']

    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 20
    com_threshold = 15
    reward = EgalitarianTheilReward(census_data=census_data,
                                    com_threshold=com_threshold)

    episodes = 10_000
    batch_size = 512
    replay_memory_size = 8192
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 5_000
    static_eps_steps = budget * 5_000

    target_network_update_step = 100

    # seed = 1033
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    q_learner = DeepMaxQLearner(base_graph=g, reward=reward, budget=budget, edge_types=edge_types,
                                target_network_update_step=target_network_update_step,
                                episodes=episodes, batch_size=batch_size,
                                replay_memory_size=replay_memory_size,
                                eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                                static_eps_steps=static_eps_steps)

    rewards_over_episodes, eps_values_over_steps = q_learner.train()
    rewards, edges = q_learner.inference()

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    sub_sampled_policy_net_loss = q_learner.policy_net_loss[0::budget]

    ax[0].plot(range(len(sub_sampled_policy_net_loss)), sub_sampled_policy_net_loss, label='Policy Net Loss')

    ax[1].plot(range(len(rewards_over_episodes)), rewards_over_episodes, label='Cumulative Reward')
    ax2 = ax[1].twinx()
    ax2.plot(range(len(eps_values_over_steps)), eps_values_over_steps, color='orange', label='Epsilon')
    fig.legend()
    plt.savefig(f'./output_{datetime.datetime.now()}.png')

    # # Plot the policy
    # _, _ = PolicyPlotter().from_model(model=q_learner.policy_net, budget=budget, actions=q_learner.actions.tolist())
    #
    # plt.show()
    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.savefig(f'./output_{datetime.datetime.now()}.png')
    # plt.show()
    logger.info(f"Removed edges: {edges}")
    q_learner.save_model(f"./ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}.pkl")
