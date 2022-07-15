from matplotlib import pyplot as plt
from typing import List


def plot_nn_loss_reward_epsilon(policy_net_loss: List[float], rewards_over_episodes: List[float],
                                eps_values_over_steps: List[float], fig: plt.Figure = None, ax: plt.Axes = None):
    """

    Args:
        policy_net_loss:
        rewards_over_episodes:
        eps_values_over_steps:
        fig:
        ax:

    Returns:

    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].plot(range(len(policy_net_loss)), policy_net_loss, label='policy net loss')
    ax[0].legend()

    lns1 = ax[1].plot(range(len(rewards_over_episodes)), rewards_over_episodes, label='maximum reward')
    ax2 = ax[1].twinx()
    lns2 = ax2.plot(range(len(eps_values_over_steps)), eps_values_over_steps, color='orange', label='epsilon')

    # Generate legend. See https://stackoverflow.com/a/5487005 for reference
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax[1].legend(lns, labs, loc=0)

    ax[0].set_ylabel('MSE Loss')
    ax[1].set_ylabel('Maximum reward')
    ax[1].set_xlabel('Episodes')
    ax2.set_ylabel('Epsilon')

    fig.tight_layout()

    return fig, ax
