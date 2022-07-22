from eptnr.baselines.optimal_baseline import optimal_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from eptnr.rewards import (
    EgalitarianTheilReward,
)
import logging
from matplotlib import pyplot as plt
from eptnr.constants.travel_metric import TravelMetric
from eptnr.plotting.solution_plotting import plot_rewards_and_graphs
import torch
import random
from eptnr import plotting

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 1
    g: ig.Graph = ig.load(f"../base_data/graph_{dataset}.gml")
    census_data = gpd.read_file(f"../base_data/census_data_{dataset}.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    num_edges = len(g.es.select(type_in=edge_types))
    # Dataset 3
    # 8  [72, 73, 74, 75, 76, 78, 81, 82]
    # 9  [72, 73, 74, 75, 76, 78, 81, 82, 79]
    # 10 [[72, 73, 74, 75, 76, 78, 81, 82, 79, 77],
    #     [72, 73, 74, 75, 76, 78, 81, 82, 79, 80]]
    budget = 3

    com_threshold = 15
    considered_metrics = [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=com_threshold,
                                    metrics=considered_metrics)

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    optimal_solutions = optimal_baseline(g=g, edge_types=edge_types, budget=budget,
                                         reward=reward)

    plot_title = f'All optimal egalitarian solutions for {budget} removals'

    # Y_Ticks:
    # Dataset_1 = list(np.arange(90, 102, 2))
    # Dataset_2 = list(np.arange(40, 72, 2))
    # Dataset_3 = list(np.arange(62, 76, 2))

    plot_rewards_and_graphs(g, optimal_solutions, plot_title,)
                            # xticks=list(range(0, num_edges + 1)),
                            # yticks=list(np.arange(40, 72, 2)))

    # fig.tight_layout()
    # fig.savefig(f"./plots/{budget}_{reward.__class__.__name__}_"
    #             f"{'_'.join([e.value for e in considered_metrics])}_optimal_solutions.png",
    #             bbox_inches='tight')
    plt.show()
    # plt.savefig(f'./plots/dataset_{dataset+1}_optimal_solutions_budget_{budget}.png')
