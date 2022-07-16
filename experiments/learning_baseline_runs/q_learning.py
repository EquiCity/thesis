import pickle

from ptnrue.baselines.rl.expected_q_learning_basleline import ExpectedQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)
from ptnrue.plotting.policy_plotting import PolicyPlotter
import logging
from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
from matplotlib import pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 5
    g = ig.load(f"../base_data/graph_{dataset}.gml")
    census_data = gpd.read_file(f"../base_data/census_data_{dataset}.geojson")
    reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl",'rb'))
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 3
    com_threshold = 15
    # reward = EgalitarianTheilReward(census_data=census_data, com_threshold=15)
    reward = CustomReward(reward_dict=reward_dict, census_data=census_data, com_threshold=15)
    episodes = 50

    q_learner = ExpectedQLearner(base_graph=g, reward=reward, edge_types=edge_types,
                                 budget=budget, episodes=episodes, step_size=1.0)
    rewards_over_episodes = q_learner.train(return_rewards_over_episodes=True)
    rewards, edges = q_learner.inference()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(range(len(rewards_over_episodes)), rewards_over_episodes)
    ax.set_title("Q-Learning cumulative reward over episodes")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Cumulative rewards")
    fig.tight_layout()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    title = f"Q-learning policy on dataset 2.2"
    ax = PolicyPlotter().from_dict(policy_dict=q_learner.q_values, actions=q_learner.actions, title=title,
                                   fig=fig, ax=ax)
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'figures/synth_ds_{dataset}_q_learning_policy.png')
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_q_learning_policy.png')
    plt.show()

    plot_title = f'Q Learning solution with custom reward and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'figures/synth_ds_{dataset}_q_learning_solution_and_rewards.png')
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_q_learning_solution_and_rewards.png')

    logger.info(f"Removed edges: {edges}")
    # q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
