import datetime

from experiments.baselines.rl.expected_q_learning import ExpectedQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards.egalitarian import egalitarian_theil
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
    reward_func = egalitarian_theil
    epochs = 1000

    q_learner = ExpectedQLearner(g, reward_func, census_data, edge_types, budget, epochs)
    rewards_over_epochs = q_learner.train()
    rewards, edges = q_learner.inference()

    plot_title = f'Q Learning solution with {reward_func.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    logger.info(f"Removed edges: {edges}")
    q_learner.save_model(f"models/ql_{epochs}_{reward_func.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
