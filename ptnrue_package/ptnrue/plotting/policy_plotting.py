import itertools as it

from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from typing import (
    Dict,
    List,
    Tuple,
    Any,
)
import torch


class PolicyPlotter:

    def __init__(self):
        pass

    def _plot_policy(self, states: List[Any], actions: List[int], policy: np.array,
                     fig: plt.Figure = None, ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        n_actions = len(actions)
        n_states = len(states)

        if not ax:
            fig, ax = plt.subplots(1, 1)

        # Adapted from https://stackoverflow.com/a/57337615
        centers = [0, n_actions - 1, 0, n_states - 1]
        dx, = np.diff(centers[:2]) / (n_actions - 1)
        dy, = -np.diff(centers[2:]) / (n_states - 1)
        extent = [centers[0] - dx / 2, centers[1] + dx / 2, centers[2] + dy / 2, centers[3] - dy / 2]
        im = ax.imshow(policy, aspect='auto', interpolation=None, extent=extent)

        for i in range(n_states):
            for j in range(n_actions):
                background_c = np.array(im.cmap(im.norm(policy[n_states - 1 - i, j]))[:3])
                text = ax.text(j, i, policy[n_states - 1 - i][j].numpy().round(1),
                               ha="center", va="center", color="black" if np.any(background_c > 0.8) else "w")

        ax.set_xlabel("Actions")
        # ax.set_xlabel()
        ax.set_ylabel("States")

        # Major ticks
        ax.set_xticks(np.arange(centers[0], centers[1] + dx, dx))
        ax.set_yticks(np.arange(centers[3], centers[2] + dy, dy))

        # Labels for major ticks
        ax.set_xticklabels(actions)
        ax.set_yticklabels(states)

        fig.colorbar(im, orientation='vertical')
        fig.tight_layout()

        return fig, ax

    def from_dict(self, policy_dict: Dict[str, np.array], actions: List[int],
                  fig: plt.Figure = None, ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        states = list(policy_dict.keys())
        states.reverse()
        n_states = len(states)
        n_actions = len(actions)

        # Making sure that the number of actions in the list of actions and in the
        # Q-table is the same
        assert n_actions == len(list(policy_dict.values())[0])

        # Initializing the policy to negative inf to identify if something
        # went wrong immediately
        policy = -np.ones((n_states, n_actions)) * np.inf

        for i, state in enumerate(states):
            policy[i] = policy_dict[state]

        return self._plot_policy(fig=fig, ax=ax, states=states,
                                 actions=actions, policy=policy)

    def from_model(self, model: nn.Module, budget: int, actions: List[int],
                   fig: plt.Figure = None, ax: plt.Axes = None) -> Tuple[plt.Figure, plt.Axes]:
        # Create states
        states = []
        possible_removed_edges = [list(combination) for k in range(0, budget + 1)
                                  for combination in it.combinations(np.arange(0, len(actions)), k)]
        for mask in possible_removed_edges:
            a = np.zeros(len(actions))
            a[mask] = 1
            states.append(torch.tensor(a, dtype=torch.float))

        states.reverse()

        policy = -torch.ones((len(states), len(actions))) * np.inf

        for i, state in enumerate(states):
            with torch.no_grad():
                policy[i] = model.forward(state)

        # Masking
        states_t = torch.cat(states).view(-1, len(actions)).bool()
        masker_not_allowed = ~states_t

        masker_end_of_budget = torch.ones(states_t.shape)
        masker_end_of_budget[states_t.sum(1) == budget] = 0

        policy = policy * masker_not_allowed
        policy[policy < 0] = 0
        policy = policy * masker_end_of_budget

        return self._plot_policy(fig=fig, ax=ax, states=states,
                                 actions=actions, policy=policy)
