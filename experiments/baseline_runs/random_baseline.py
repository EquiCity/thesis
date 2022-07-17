from ptnrue.baselines import random_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import EgalitarianTheilReward
import logging
import random
import torch

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("../base_data/graph_1.gml")
    census_data = gpd.read_file("../base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    logger.info(f"Considered edges: {edge_types}")
    budget = 3
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=15)

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    rewards, edges = random_baseline(g=g, reward=reward,
                                     edge_types=edge_types, budget=budget)
    logger.info(f"Removed edges: {edges}")
