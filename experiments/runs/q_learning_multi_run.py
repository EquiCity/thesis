import datetime

from experiments.baselines.rl.expected_q_learning import ExpectedQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards import (
    egalitarian_theil,
    utilitarian,
    simple,
)
import logging
from experiments.plotting.solution_plotting import plot_rewards_and_graphs
from matplotlib import pyplot as plt
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    reward_func = simple
    epochs = 1000
    runs = 100
    random_seeds = np.arange(0,runs)+2048

    successful = 0

    for r in tqdm(range(runs)):
        np.random.seed(random_seeds[r])
        q_learner = ExpectedQLearner(g, reward_func, census_data, edge_types, budget, epochs, step_size=1)
        rewards_over_epochs = q_learner.train(return_rewards_over_epochs=True, verbose=False)
        rewards, edges = q_learner.inference()
        if set(edges) == {72, 73, 74, 75, 76, 77, 78, 79, 82} or \
            set(edges) == {72, 73, 74, 75, 76, 77, 78, 79, 81}:
            successful += 1

    logger.info(f"Finds optimum in {successful} out of {runs}; {successful*100/runs}")
