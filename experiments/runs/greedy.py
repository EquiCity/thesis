from experiments.baselines.greedy import greedy_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards.egalitarian import egalitarian_theil
import logging
from matplotlib import pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__=="__main__":
    g = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    reward_func = egalitarian_theil

    # Optimal: [72, 73, 74, 75, 76, 77, 78, 81, 82]
    # Greedy: [82, 81, 78, 76, 75, 74, 73, 72, 72]
    rewards, edges = greedy_baseline(g, census_data, edge_types, budget, reward_func)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.show()
    logger.info(f"Removed edges: {[e.index for e in edges]}")
