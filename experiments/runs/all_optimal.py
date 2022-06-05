import matplotlib.patches

from experiments.baselines.optimal import optimal_baseline
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards import (
    egalitarian_jsd,
    egalitarian_theil,
    utilitarian,
    elitarian,
)
import logging
from matplotlib import pyplot as plt
from experiments.constants.travel_metric import TravelMetric
from experiments.plotting.subplots_formatting import fixed_ax_aspect_ratio

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g: ig.Graph = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')

    budget = 9

    reward_func = egalitarian_jsd
    considered_metrics = [TravelMetric.TT]#, TravelMetric.HOPS, TravelMetric.COM]

    optimal_solutions = optimal_baseline(g=g, census_data=census_data, edge_types=edge_types, budget=budget,
                                      reward_func=reward_func, metrics=considered_metrics,
                                      com_threshold=15)

    fig, ax = plt.subplots(len(optimal_solutions), 2, figsize=(10, 40*len(optimal_solutions)//8))
    fig.suptitle(f'All optimal {reward_func.__name__} solutions for budget size {budget}', fontsize=20)

    for i, (rewards, edges) in enumerate(optimal_solutions):

        reward_plot = ax[0] if len(optimal_solutions) == 1 else ax[i][0]
        graph_plot = ax[1] if len(optimal_solutions) == 1 else ax[i][1]

        edges_idxs = [e.index for e in edges]

        reward_plot.plot(np.arange(len(rewards)), rewards, '--bo')

        major_ticks_top = np.linspace(0,np.min(rewards), 10)
        minor_ticks_top = np.linspace(0,np.min(rewards), 100)

        # ax[i][0].set_yticks(major_ticks_top)
        # ax[i][0].set_yticks(minor_ticks_top, minor=True)

        reward_plot.grid(which="major", alpha=0.6)
        reward_plot.grid(which="minor", alpha=0.3)

        g_prime = g.copy()
        g_prime.es['color'] = 'red'
        fixed_ax_aspect_ratio(graph_plot, 1)
        ig.plot(g_prime, target=graph_plot)
        arrows = [e for e in graph_plot.get_children() if
                  isinstance(e, matplotlib.patches.FancyArrowPatch)]  # This is a PathCollection

        label_set = False
        for j, (arrow, edge) in enumerate(zip(arrows, g.es)):

            if edge['type'] == 'walk':
                arrow.set_color('gray')
                arrow.set_alpha(0.2)
            elif edge.index in edges_idxs:
                arrow.set_color('tomato')
                arrow.set_linewidth(3)
                # Make sure label is only set once
                if not label_set:
                    arrow.set_label('removed')
                    label_set = True

        # ig.plot(g.subgraph_edges(edges), palette='viridis', target=ax[1])
        graph_plot.legend()

        logger.info(f"For optimal sol {i}, Removed edges: {[e.index for e in edges]}")
    # fig.tight_layout()
    fig.savefig(f"./plots/{budget}_{reward_func.__name__}_"
                f"{'_'.join([e.value for e in considered_metrics])}_optimal_solutions.png",
                bbox_inches='tight')
