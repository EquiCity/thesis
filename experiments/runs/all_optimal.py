import matplotlib.patches

from experiments.baselines.optimal import optimal_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards import (
    egalitarian_jsd,
    egalitarian_theil,
    utilitarian,
    elitarian,
)
import logging
from matplotlib import pyplot as plt
from experiments.constants.travel_metric import TravelMetric
from experiments.plotting.solution_plotting import plot_rewards_and_graphs

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g: ig.Graph = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    budget = 9

    reward_func = egalitarian_theil
    considered_metrics = [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]

    optimal_solutions = optimal_baseline(g=g, census_data=census_data, edge_types=edge_types, budget=budget,
                                         reward_func=reward_func, metrics=considered_metrics,
                                         com_threshold=15)

    plot_title = f'All optimal {reward_func.__name__} solutions for budget size {budget}'
    plot_rewards_and_graphs(g, optimal_solutions, plot_title)

    # fig.tight_layout()
    # fig.savefig(f"./plots/{budget}_{reward_func.__name__}_"
    #             f"{'_'.join([e.value for e in considered_metrics])}_optimal_solutions.png",
    #             bbox_inches='tight')
    plt.show()
