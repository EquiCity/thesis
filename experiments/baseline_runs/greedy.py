import matplotlib.pyplot as plt

from eptnr.baselines.greedy_baseline import greedy_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from eptnr.rewards import EgalitarianTheilReward
import logging
from eptnr.plotting.solution_plotting import plot_rewards_and_graphs
import torch
import random

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("../base_data/graph_1.gml")
    census_data = gpd.read_file("../base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 3
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=15)

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rewards, edges = greedy_baseline(g=g, reward=reward, edge_types=edge_types, budget=budget, )
    plot_title = f'Greedy {reward.__class__.__name__} solution for budget size {budget}'
    plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()
    logger.info(f"Removed edges: {edges}")
