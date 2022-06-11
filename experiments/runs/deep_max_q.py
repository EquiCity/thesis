import datetime

from experiments.deep_q_learning_approach.dense_d_max_ql import DeepMaxQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards import (
    EgalitarianTheilReward,
)
import logging
from experiments.plotting.solution_plotting import plot_rewards_and_graphs
from matplotlib import pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    com_threshold = 15
    reward = EgalitarianTheilReward(census_data, com_threshold)
    episodes = 1000

    q_learner = DeepMaxQLearner(g, reward, edge_types, budget, episodes)
    rewards_over_episodes = q_learner.train()
    rewards, edges = q_learner.inference()

    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    logger.info(f"Removed edges: {edges}")
    q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
