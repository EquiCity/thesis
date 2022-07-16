import datetime
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
from ptnrue.plotting.policy_plotting import PolicyPlotter
from matplotlib import pyplot as plt
from pathlib import Path

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 5
    g = ig.load(Path(f"../base_data/graph_{dataset}.gml"))
    census_data = gpd.read_file(Path(f"../base_data/census_data_{dataset}.geojson"))
    reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 3
    com_threshold = 15
    reward = CustomReward(reward_dict=reward_dict, census_data=census_data,
                          com_threshold=com_threshold)

    episodes = 150

    # Replay Memory
    batch_size = 32
    replay_memory_size = 256
    target_network_update_step = 20

    # EPS Schedule
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 50
    static_eps_steps = 50 * budget

    seed = 1024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # deep_max_q_learner = DeepMaxQLearner.load_model(Path('model_snapshots/model_250_2022-07-15T11:03:11.648540.pkl'))
    deep_max_q_learner = DeepMaxQLearner(base_graph=g, reward=reward, budget=budget, edge_types=edge_types,
                                         target_network_update_step=target_network_update_step,
                                         episodes=episodes, batch_size=batch_size,
                                         replay_memory_size=replay_memory_size,
                                         eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                                         static_eps_steps=static_eps_steps)

    cum_rewards_over_episodes, max_rewards_over_episodes, eps_values_over_episodes = deep_max_q_learner.train()
    rewards, edges = deep_max_q_learner.inference()

    sub_sampled_policy_net_loss = deep_max_q_learner.policy_net_loss[0::budget]
    sub_sampled_policy_net_loss = np.concatenate([np.repeat(sub_sampled_policy_net_loss[0], batch_size // budget),
                                                  sub_sampled_policy_net_loss])
    title = "Deep MaxQ-learning network loss\nand policy performance"
    fig, ax = plot_nn_loss_reward_epsilon(sub_sampled_policy_net_loss,
                                          {
                                              'maximum reward': max_rewards_over_episodes,
                                              'cumulative reward': cum_rewards_over_episodes,
                                          },
                                          eps_values_over_episodes, title=title)

    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'figures/synth_ds_{dataset}_deep_max_q_learning_behavior_{episodes}.png')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/'
    #     f'overleaf/62a466789b2183065a639cda/content-media/'
    #     f'synth_ds_{dataset}_deep_max_q_learning_behavior_{episodes}.png')
    plt.show()

    # Plot the policy
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    title = f"Deep MaxQ-learning policy on dataset 2.2"
    _, _ = PolicyPlotter().from_model(model=deep_max_q_learner.policy_net, budget=budget,
                                      actions=deep_max_q_learner.actions.tolist(),
                                      title=title,
                                      fig=fig, ax=ax)
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'figures/synth_ds_{dataset}_deep_max_q_learning_policy.png')
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_deep_max_q_learning_policy.png')
    plt.show()
    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()
    # logger.info(f"Removed edges: {edges}")
    # deep_max_q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
