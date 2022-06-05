import matplotlib.patches

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
from matplotlib import pyplot as plt
from experiments.constants.travel_metric import TravelMetric

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g: ig.Graph = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    budget = 1
    reward_func = utilitarian

    optimal_solutions = optimal_baseline(g=g, census_data=census_data, edge_types=edge_types, budget=budget,
                                      reward_func=reward_func, metrics=[TravelMetric.TT],
                                      com_threshold=15)
    rewards, edges = optimal_solutions[0]

    edges_idxs = [e.index for e in edges]
    fig, ax = plt.subplots(1, 2)

    ax[0].plot(np.arange(len(rewards)), rewards, '--bo')
    g_prime = g.copy()
    g_prime.es['color'] = 'red'
    ig.plot(g_prime, target=ax[1])
    arrows = [e for e in ax[1].get_children() if
              isinstance(e, matplotlib.patches.FancyArrowPatch)]  # This is a PathCollection

    for arrow, edge in zip(arrows, g.es):
        if edge['type'] == 'walk':
            arrow.set_color('gray')
            arrow.set_alpha(0.2)
        elif edge.index in edges_idxs:
            arrow.set_color('tomato')
            arrow.set_linewidth(3)
            arrow.set_label('removed')

    # ig.plot(g.subgraph_edges(edges), palette='viridis', target=ax[1])
    ax[1].legend()
    plt.show()
    logger.info(f"Removed edges: {[e.index for e in edges]}")
