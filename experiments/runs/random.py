from experiments.baselines.random import random_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards.egalitarian import egalitarian
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__=="__main__":
    g = ig.load("./base_data/graph.gml")
    census_data = gpd.read_file("./base_data/census_data.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    logger.info(f"Considered edges: {edge_types}")
    budget = 5
    reward_func = egalitarian

    rewards, edges = random_baseline(g, census_data, edge_types, budget, reward_func)
    logger.info(f"Removed edges: {[e.index for e in edges]}")
