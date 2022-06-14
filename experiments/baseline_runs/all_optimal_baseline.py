from ptnrue_package.ptnrue import optimal_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue_package.ptnrue import (
    EgalitarianTheilReward,
)
import logging
from matplotlib import pyplot as plt
from ptnrue_package.ptnrue import TravelMetric
from ptnrue_package.ptnrue.plotting import plot_rewards_and_graphs

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g: ig.Graph = ig.load("../base_data/graph_1.gml")
    census_data = gpd.read_file("../base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    budget = 9

    com_threshold = 15
    considered_metrics = [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=com_threshold,
                                    metrics=considered_metrics)

    optimal_solutions = optimal_baseline(g=g, edge_types=edge_types, budget=budget,
                                         reward=reward)

    plot_title = f'All optimal {reward.__class__.__name__} solutions for budget size {budget}'
    plot_rewards_and_graphs(g, optimal_solutions, plot_title)

    # fig.tight_layout()
    # fig.savefig(f"./plots/{budget}_{reward.__class__.__name__}_"
    #             f"{'_'.join([e.value for e in considered_metrics])}_optimal_solutions.png",
    #             bbox_inches='tight')
    plt.show()
