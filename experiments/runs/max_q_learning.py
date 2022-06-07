from experiments.baselines.rl.max_q_learner import MaxQLearner
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
    epochs = 1000

    max_q_learner = MaxQLearner(g, reward_func, census_data, edge_types, budget, epochs)
    rewards_over_epochs = max_q_learner.train()
    rewards, edges = max_q_learner.inference()

    plot_title = f'Max Q Learning solution with {reward_func.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    logger.info(f"Removed edges: {edges}")
    max_q_learner.save_model(f"models/mql_{epochs}_{reward_func.__name__}_{budget}_{datetime.now()}_.pkl")
