from typing import List, Tuple, Union

import igraph as ig
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .subplots_formatting import fixed_ax_aspect_ratio
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def plot_rewards_and_graphs(base_graph: ig.Graph, solutions: List[Tuple[List[float], List[int]]],
                            title: str, xticks: List[Union[float, int]] = None,
                            yticks: List[Union[float, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(len(solutions), 2, figsize=(10, 40 * len(solutions) // 8))
    fig.suptitle(title, fontsize=20)

    for i, (rewards, edges) in enumerate(solutions):

        reward_plot = ax[0] if len(solutions) == 1 else ax[i][0]
        graph_plot = ax[1] if len(solutions) == 1 else ax[i][1]

        x = np.arange(1, len(rewards)+1)

        reward_plot.plot(x, rewards, '--bo')
        if not xticks:
            reward_plot.set_xlim([0, len(rewards)+1])
        else:
            reward_plot.set_xlim([min(xticks), max(xticks)])
            reward_plot.set_xticks(xticks)
        reward_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

        if not yticks:
            reward_plot.set_ylim([min(rewards) - 1, max(rewards) + 1])
        else:
            reward_plot.set_ylim([min(yticks), max(yticks)])
            reward_plot.set_yticks(yticks)

        # major_ticks_top = np.linspace(0, np.min(rewards), 10)
        # minor_ticks_top = np.linspace(0, np.min(rewards), 100)

        # ax[i][0].set_yticks(major_ticks_top)
        # ax[i][0].set_yticks(minor_ticks_top, minor=True)

        reward_plot.grid(which="major", alpha=0.6)
        reward_plot.grid(which="minor", alpha=0.3)

        for i, txt in enumerate(edges):
            reward_plot.annotate(txt, (x[i], rewards[i]))

        g_prime: ig.Graph = base_graph.copy()
        g_prime.vs.select(type_eq='res_node')['color'] = 'red'
        g_prime.vs.select(type_eq='pt_node')['color'] = 'blue'
        g_prime.vs.select(type_eq='poi_node')['color'] = 'green'

        color_dict = {
            'res_node': 'red',
            'pt_node': 'blue',
            'poi_node': 'green',
        }

        fixed_ax_aspect_ratio(graph_plot, 1)
        ig.plot(g_prime, target=graph_plot, vertex_size=10,
                vertex_color = [color_dict[t] for t in g_prime.vs["type"]],)
                # edge_width=4, edge_arrow_size=6, edge_arrow_width=3)
        arrows = [e for e in graph_plot.get_children() if
                  isinstance(e, matplotlib.patches.FancyArrowPatch)]  # This is a PathCollection

        label_set = False
        for j, (arrow, edge) in enumerate(zip(arrows, base_graph.es)):

            if edge['type'] == 'walk':
                arrow.set_color('gray')
                arrow.set_alpha(0.2)
            elif edge.index in edges:
                arrow.set_color('tomato')
                arrow.set_linewidth(3)
                # Make sure label is only set once
                if not label_set:
                    arrow.set_label('removed')
                    label_set = True

        logger.info(f"For solution {i}, Removed edges: {edges}")

    # ig.plot(g.subgraph_edges(edges), palette='viridis', target=ax[1])
    graph_plot.legend()

    return fig, ax
