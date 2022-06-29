from ptnrue.rewards import (
    EgalitarianTheilReward,
    UtilitarianReward,
    ElitarianReward,
)
import igraph as ig
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs

from ptnrue.baselines.optimal_baseline import optimal_baseline, optimal_baseline_up_to_budget_k
from matplotlib import pyplot as plt


def retrieve_dataset(graph_path: Path, census_path: Path) -> Tuple[ig.Graph, gpd.GeoDataFrame, List[str]]:
    g: ig.Graph = ig.load(graph_path)
    census_data = gpd.read_file(census_path)
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    return g, census_data, edge_types


if __name__ == "__main__":
    datasets = {
        "dataset_1": {
            "graph_path": Path("../base_data/graph_1.gml"),
            "census_path": Path("../base_data/census_data_1.geojson"),
        },
        "dataset_2": {
            "graph_path": Path("../base_data/graph_2.gml"),
            "census_path": Path("../base_data/census_data_2.geojson"),
        },
        "dataset_3": {
            "graph_path": Path("../base_data/graph_3.gml"),
            "census_path": Path("../base_data/census_data_3.geojson"),
        },
    }

    for dataset in datasets:
        g, census_data, edge_types = retrieve_dataset(datasets[dataset]['graph_path'], datasets[dataset]['census_path'])
        datasets[dataset]['graph'] = g
        datasets[dataset]['census_data'] = census_data
        datasets[dataset]['edge_types'] = edge_types

        max_budget = len(g.es.select(type_in=edge_types))
        datasets[dataset]['max_budget'] = max_budget

        datasets[dataset]['rewards'] = {}

        datasets[dataset]['rewards']['egalitarian'] = EgalitarianTheilReward(census_data=census_data, com_threshold=15)
        datasets[dataset]['rewards']['utilitarian'] = UtilitarianReward(census_data=census_data, com_threshold=15)
        datasets[dataset]['rewards']['elitarian'] = ElitarianReward(census_data=census_data, com_threshold=15,
                                                                    groups=['purple'])

        datasets[dataset]['configuration'] = {}

        for reward_func in datasets[dataset]['rewards']:
            datasets[dataset]['configuration'][reward_func] = optimal_baseline_up_to_budget_k(
                g=datasets[dataset]['graph'],
                reward=datasets[dataset]['rewards'][reward_func],
                edge_types=datasets[dataset]['edge_types'],
                budget=datasets[dataset]['max_budget']
            )

            plot_rewards_and_graphs(datasets[dataset]['graph'],
                                    datasets[dataset]['configuration'][reward_func],
                                    f"Optimal solution configuration {reward_func} for {dataset}")
            plt.show()


