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
from ptnrue.plotting.deep_solution_plotting import plot_nn_loss_reward_epsilon
from matplotlib import pyplot as plt
from pathlib import Path
import ray

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

# ray.init(ignore_reinit_error=True)

GRAPH_PATH = Path(os.environ['GRAPH_PATH'])
CENSUS_PARQUET_PATH = Path(os.environ['CENSUS_PARQUET_PATH'])

if __name__ == "__main__":
    logger.info("Loading data")
    g: ig.Graph = ig.load(GRAPH_PATH)
    census_data = gpd.read_parquet(CENSUS_PARQUET_PATH)
    logger.info("Completed loading data")

    census_data["neighborhood"] = 'RC_' + census_data['BU_NAAM']
    census_data['n_inh'] = census_data['a_inw']
    census_data["res_centroids"] = census_data["res_centroid"]
    census_data["geometry"] = census_data["res_centroid"]

    all_w_nw_inh = census_data['a_w_all'] + census_data['a_nw_all'] + 0.0001

    census_data["n_western"] = (census_data['a_inw'] * (census_data['a_w_all'] / all_w_nw_inh)).astype('int')
    census_data["n_non_western"] = census_data['a_inw'] - census_data["n_western"]

    census_data = census_data[["neighborhood", "n_inh", "n_western", "n_non_western", "res_centroids", "geometry"]]

    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 10
    com_threshold = 15
    reward = EgalitarianTheilReward(census_data=census_data,
                                    com_threshold=com_threshold)

    episodes = 800
    batch_size = 256
    replay_memory_size = 4096
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 1000
    static_eps_steps = budget * 500

    target_network_update_step = 50

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    ams_deep_max_q_learner = DeepMaxQLearner.load_model(Path('model_snapshots/model_750_2022-07-15T13:09:14.687741.pkl'))
    # ams_deep_max_q_learner = DeepMaxQLearner(base_graph=g, reward=reward, budget=budget, edge_types=edge_types,
    #                                          target_network_update_step=target_network_update_step,
    #                                          episodes=episodes, batch_size=batch_size,
    #                                          replay_memory_size=replay_memory_size,
    #                                          eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
    #                                          static_eps_steps=static_eps_steps)

    logger.info("Starting Training")
    cum_rewards_over_episodes, max_rewards_over_episodes, eps_values_over_episodes = ams_deep_max_q_learner.train()
    ams_deep_max_q_learner.save_model(f'model_{episodes}_2022-07-15T13:09:14.687741.pkl')

    rewards, edges = ams_deep_max_q_learner.inference()

    sub_sampled_policy_net_loss = ams_deep_max_q_learner.policy_net_loss[0::budget]
    fig, ax = plot_nn_loss_reward_epsilon(sub_sampled_policy_net_loss, max_rewards_over_episodes,
                                          eps_values_over_episodes)

    fig.savefig(f'./ams_deep_max_q_learning_behavior_{episodes}.svg')
    fig.savefig(f'./ams_deep_max_q_learning_behavior_{episodes}.png')

    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    fig.savefig('./solution_graphs.png')
    fig.savefig('./solution_graphs.svg')

    logger.info(f"Removed edges: {edges} | rewards: {rewards}")
