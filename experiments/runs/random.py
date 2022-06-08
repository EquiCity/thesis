from experiments.baselines.random import random_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards.egalitarian import egalitarian_theil
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    logger.info(f"Considered edges: {edge_types}")
    budget = 9
    reward_func = egalitarian_theil

    rewards, edges = random_baseline(g, census_data, edge_types, budget, reward_func)
    logger.info(f"Removed edges: {edges}")
