from ptnrue.baselines.ga_baseline import ga_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue import EgalitarianTheilReward
from ptnrue.plotting import plot_rewards_and_graphs
import logging
from matplotlib import pyplot as plt

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("../base_data/graph_1.gml")
    census_data = gpd.read_file("../base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=15)

    reward, edges = ga_baseline(g=g, reward=reward, edge_types=edge_types,
                                budget=budget, num_generations=100, sol_per_pop=100,
                                num_parents_mating=20, saturation=20,
                                mutation_probability=0.5, crossover_probability=0.5)
    plot_rewards_and_graphs(base_graph=g, solutions=[(reward, edges),], title=f"GA solution for budget size {budget}")
    plt.show()
    logger.info(f"Removed edges: {edges}")
