from experiments.baselines.ga import ga_baseline
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
    budget = 5
    reward_func = egalitarian_theil

    rewards, edges = ga_baseline(g, census_data, edge_types, budget, reward_func, num_generations=20000)
    plt.plot(np.arange(rewards.size), rewards)
    plt.show()
    logger.info(f"Removed edges: {edges}")
