from ptnrue.baselines.rl.max_q_learner_baseline import MaxQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)
from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
import logging
from ptnrue.plotting.policy_plotting import PolicyPlotter
import pickle
from matplotlib import pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 5
    g = ig.load(f"../base_data/graph_{dataset}.gml")
    census_data = gpd.read_file(f"../base_data/census_data_{dataset}.geojson")
    reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 3
    com_threshold = 15
    reward = CustomReward(reward_dict=reward_dict, census_data=census_data,
                          com_threshold=com_threshold)
    episodes = 50

    max_q_learner = MaxQLearner(g, reward, edge_types, budget, episodes)
    rewards_over_episodes = max_q_learner.train(return_rewards_over_episodes=True)
    rewards, edges = max_q_learner.inference()

    plt.plot(range(len(rewards_over_episodes)), rewards_over_episodes)
    plt.title("Q Learning rewards over episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Return)")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax = PolicyPlotter().from_dict(policy_dict=max_q_learner.q_values, actions=max_q_learner.actions,
                                   fig=fig, ax=ax)
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'figures/synth_ds_{dataset}_max_q_learning_policy.png')
    fig.savefig(
        f'/home/rico/Documents/thesis/paper/'
        f'overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_max_q_learning_policy.png')
    plt.show()

    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    logger.info(f"Removed edges: {edges}")
    # q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
