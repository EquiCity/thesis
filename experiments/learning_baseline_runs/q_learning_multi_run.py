from ptnrue_package.ptnrue.baselines.rl.expected_q_learning_basleline import ExpectedQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue_package.ptnrue.rewards import (
    EgalitarianTheilReward
)
from ptnrue_package.ptnrue.baselines.optimal_baseline import optimal_baseline
import logging
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("../base_data/graph_1.gml")
    census_data = gpd.read_file("../base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    com_threshold = 15
    reward = EgalitarianTheilReward(census_data, com_threshold)
    episodes = 100
    runs = 10
    random_seeds = np.arange(0,runs)+2048

    successful = 0

    logger.info("Finding optimal solutions first")
    optimal_solutions = optimal_baseline(g=g, edge_types=edge_types,
                                         budget=budget, reward=reward)
    optimal_edges = [set(opt[1]) for opt in optimal_solutions]

    logger.info("Proceeding with multi-run Q-Learning")
    for r in tqdm(range(runs)):
        np.random.seed(random_seeds[r])
        q_learner = ExpectedQLearner(g, reward, edge_types, budget, episodes, step_size=0.1)
        rewards_over_episodes = q_learner.train(return_rewards_over_episodes=True, verbose=False)
        rewards, edges = q_learner.inference()
        edge_diffs = [set(edges).symmetric_difference(opt) for opt in optimal_edges]
        if any([ed == set() for ed in edge_diffs]):
            logger.info("Found optimal solution")
            successful += 1
        else:
            for diff in edge_diffs:
                logger.info(f"\nSolution {r}:\t{edges}\n"
                            f"Optimal solutions:\t{optimal_edges}\n"
                            f"Difference:\t{diff}")

    logger.info(f"Finds optimum in {successful} out of {runs}; {successful*100/runs}%")
