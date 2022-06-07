from typing import List, Tuple

import igraph as ig
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from .subplots_formatting import fixed_ax_aspect_ratio
import logging

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def plot_rewards_and_graphs(base_graph: ig.Graph, solutions: List[Tuple[List[float], List[int]]],
                            title: str) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(len(solutions), 2, figsize=(10, 40 * len(solutions) // 8))
    fig.suptitle(title, fontsize=20)

    for i, (rewards, edges) in enumerate(solutions):

        reward_plot = ax[0] if len(solutions) == 1 else ax[i][0]
        graph_plot = ax[1] if len(solutions) == 1 else ax[i][1]

        reward_plot.plot(np.arange(len(rewards)), rewards, '--bo')

        # major_ticks_top = np.linspace(0, np.min(rewards), 10)
        # minor_ticks_top = np.linspace(0, np.min(rewards), 100)

        # ax[i][0].set_yticks(major_ticks_top)
        # ax[i][0].set_yticks(minor_ticks_top, minor=True)

        reward_plot.grid(which="major", alpha=0.6)
        reward_plot.grid(which="minor", alpha=0.3)

        g_prime = base_graph.copy()
        fixed_ax_aspect_ratio(graph_plot, 1)
        ig.plot(g_prime, target=graph_plot)
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
