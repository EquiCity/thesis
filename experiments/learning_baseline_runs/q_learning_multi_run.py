import pickle

from ptnrue.baselines.rl.expected_q_learning_basleline import ExpectedQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)
from ptnrue.baselines.run_utils.multi_run import multi_run
from ptnrue.baselines.optimal_baseline import optimal_max_baseline
import logging
from tqdm import tqdm
import torch
import random
from ptnrue import plotting

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 5
    g = ig.load(f"../base_data/graph_{dataset}.gml")
    census_data = gpd.read_file(f"../base_data/census_data_{dataset}.geojson")
    reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 3
    com_threshold = 15
    reward = CustomReward(reward_dict=reward_dict, census_data=census_data, com_threshold=com_threshold)
    episodes = 150
    n_runs = 10

    seed = 2048
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    opt_edges = optimal_max_baseline(g=g, reward=reward, edge_types=edge_types, budget=budget)
    optimal_edges = [o[1] for o in opt_edges]

    def learning_algo():
        q_learner = ExpectedQLearner(g, reward, edge_types, budget, episodes, step_size=1)
        q_learner.train(verbose=False)
        return q_learner.inference()

    multi_run(algo=learning_algo, n_runs=n_runs, optimal_edges=optimal_edges, verbose=True)
