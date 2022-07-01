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


def plot_rewards(ax: plt.Axes, edges: List[int], rewards: List[float],
                 xticks: List[Union[int, float]], yticks: List[Union[int, float]]):
    x = np.arange(1, len(rewards) + 1)

    ax.plot(x, rewards, '--bo')
    if not xticks:
        ax.set_xlim([0, len(rewards) + 1])
    else:
        ax.set_xlim([min(xticks), max(xticks)])
        ax.set_xticks(xticks)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if not yticks:
        ax.set_ylim([min(rewards) - 1, max(rewards) + 1])
    else:
        ax.set_ylim([min(yticks), max(yticks)])
        ax.set_yticks(yticks)

    # major_ticks_top = np.linspace(0, np.min(rewards), 10)
    # minor_ticks_top = np.linspace(0, np.min(rewards), 100)

    # ax[i][0].set_yticks(major_ticks_top)
    # ax[i][0].set_yticks(minor_ticks_top, minor=True)

    ax.grid(which="major", alpha=0.6)
    ax.grid(which="minor", alpha=0.3)

    for i, txt in enumerate(edges):
        ax.annotate(txt, (x[i], rewards[i]))

    return ax


def plot_graph(ax: plt.Axes, base_graph: ig.Graph, edges: List[int]):
    g_prime: ig.Graph = base_graph.copy()
    g_prime.vs.select(type_eq='res_node')['color'] = 'red'
    g_prime.vs.select(type_eq='pt_node')['color'] = 'blue'
    g_prime.vs.select(type_eq='poi_node')['color'] = 'green'

    color_dict = {
        'res_node': 'red',
        'pt_node': 'blue',
        'poi_node': 'green',
    }

    fixed_ax_aspect_ratio(ax, 1)
    ig.plot(g_prime, target=ax, vertex_size=10,
            vertex_color=[color_dict[t] for t in g_prime.vs["type"]], )
    # edge_width=4, edge_arrow_size=6, edge_arrow_width=3)
    arrows = [e for e in ax.get_children() if
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

    return ax


def plot_rewards_and_graphs(base_graph: ig.Graph, solutions: List[Tuple[List[float], List[int]]],
                            title: str, xticks: List[Union[float, int]] = None,
                            yticks: List[Union[float, int]] = None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(len(solutions), 2, figsize=(10, 40 * len(solutions) // 8))
    fig.suptitle(title, fontsize=20)

    for i, (rewards, edges) in enumerate(solutions):

        reward_plot = ax[0] if len(solutions) == 1 else ax[i][0]
        graph_plot = ax[1] if len(solutions) == 1 else ax[i][1]

        _ = plot_rewards(reward_plot, edges, rewards, xticks, yticks)
        graph_plot = plot_graph(graph_plot, base_graph, edges)
        graph_plot.legend()

        logger.info(f"For solution {i}, Removed edges: {edges}")

    return fig, ax
