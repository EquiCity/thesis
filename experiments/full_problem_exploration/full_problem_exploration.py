import matplotlib.pyplot as plt

from ptnrue.analysis.full_problem_exploration import full_problem_exploration
from ptnrue.plotting.solution_plotting import plot_full_problem_exploration
import igraph as ig
import pickle
import geopandas as gpd
from pathlib import Path
import numpy as np
from ptnrue.rewards import (
    EgalitarianTheilReward,
    CustomReward,
)


if __name__ == '__main__':
    dataset = 2

    g = ig.load(Path(f"../base_data/graph_{dataset}.gml"))
    census_data = gpd.read_file(Path(f"../base_data/census_data_{dataset}.geojson"))
    # reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))

    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    # g.es[[78, 81, 76, 74, 82, 73, 72, 75]]['active'] = 0

    com_threshold = 15
    # reward = CustomReward(reward_dict=reward_dict, census_data=census_data,
    #                       com_threshold=com_threshold)
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=com_threshold)

    configurations, rewards = full_problem_exploration(g, reward, edge_types)
    fig, axs = plot_full_problem_exploration(base_graph=g, configurations=configurations, rewards=rewards)
    plt.show()

    # fig.savefig(f'/home/rico/Documents/thesis/paper/'
    #             f'overleaf/62a466789b2183065a639cda/content-media/'
    #             f'all_configurations_ds_{dataset}_{reward.__class__.__name__}_BASE.svg')
    # fig.savefig(f'./plots/all_configurations_ds_{dataset}_{reward.__class__.__name__}.png')
