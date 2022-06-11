from experiments.baselines.rl.sarsa_learner import SARSALearner
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards.egalitarian import egalitarian_theil
import logging
from experiments.plotting.solution_plotting import plot_rewards_and_graphs
from matplotlib import pyplot as plt
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    reward_func = egalitarian_theil
    episodes = 1000

    q_learner = SARSALearner(g, reward_func, census_data, edge_types, budget, episodes)
    rewards_over_episodes = q_learner.train(return_rewards_over_episodes=True)
    rewards, edges = q_learner.inference()

    plt.plot(range(len(rewards_over_episodes)), rewards_over_episodes)
    plt.title("SARSA rewards over episodes")
    plt.xlabel("Reward")
    plt.ylabel("episodes")
    plt.show()

    # plot_title = f'SARSA solution with {reward.__class__.__name__} and budget size {budget}'
    # fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    # plt.show()

    logger.info(f"Removed edges: {edges}")
    # q_learner.save_model(f"models/sarsa_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.now()}.pkl")
