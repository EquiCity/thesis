import matplotlib.patches
import matplotlib.pyplot as plt

from experiments.baselines.optimal import optimal_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards import (
    egalitarian_theil,
    utilitarian,
    elitarian,
)
import logging
from experiments.plotting.solution_plotting import plot_rewards_and_graphs
from experiments.constants.travel_metric import TravelMetric

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g: ig.Graph = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    budget = 10
    reward_func = egalitarian_theil

    optimal_solutions = optimal_baseline(g=g, census_data=census_data, edge_types=edge_types, budget=budget,
                                         reward_func=reward_func, metrics=[TravelMetric.TT],
                                         com_threshold=15)
    rewards, edges = optimal_solutions[0]

    plot_title = f'Optimal solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()
