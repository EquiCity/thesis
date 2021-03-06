import matplotlib.pyplot as plt

from eptnr.baselines import optimal_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from eptnr.rewards import (
    EgalitarianTheilReward,
)
import logging
from eptnr.plotting.solution_plotting import plot_rewards_and_graphs
import torch
import random

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g: ig.Graph = ig.load("../base_data/graph_1.gml")
    census_data = gpd.read_file("../base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    budget = 3
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=15)

    optimal_solutions = optimal_baseline(g=g, edge_types=edge_types,
                                         budget=budget, reward=reward)

    rewards, edges = optimal_solutions[0]

    plot_title = f'Optimal solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()
