import datetime
import pickle

from ptnrue.deep_q_learning_approach.deep_q_learner import DeepQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)
import logging
from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
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

    episodes = 10_000
    batch_size = 128
    replay_memory_size = 512
    eps_start = 1.0
    eps_end = 0.0
    eps_decay = 100
    static_eps_steps = 6000

    target_network_update_step = 500

    q_learner = DeepQLearner(base_graph=g, reward=reward, budget=budget, edge_types=edge_types,
                             target_network_update_step=target_network_update_step,
                             episodes=episodes, batch_size=batch_size,
                             replay_memory_size=replay_memory_size,
                             eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                             static_eps_steps=static_eps_steps)

    rewards_over_episodes = q_learner.train()
    rewards, edges = q_learner.inference()

    fig,ax = plt.subplots(2, 1, figsize=(10,10))
    ax[0].plot(range(len(q_learner.network_loss)), q_learner.network_loss)

    ax[1].plot(range(len(rewards_over_episodes)), rewards_over_episodes)
    plt.show()

    # Plot the policy
    _, _ = PolicyPlotter().from_model(model=q_learner.policy_net, budget=budget, actions=q_learner.actions.tolist())

    plt.show()
    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()
    # logger.info(f"Removed edges: {edges}")
    # q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
