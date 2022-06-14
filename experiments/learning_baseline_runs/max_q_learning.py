from ptnrue_package.ptnrue.baselines.rl.max_q_learner_baseline import MaxQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue_package.ptnrue.rewards import (
    EgalitarianTheilReward
)
from ptnrue_package.ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
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
    com_threshold = 15
    reward = EgalitarianTheilReward(census_data, com_threshold)
    episodes = 100

    max_q_learner = MaxQLearner(g, reward, edge_types, budget, episodes)
    rewards_over_episodes = max_q_learner.train(return_rewards_over_episodes=True)
    rewards, edges = max_q_learner.inference()

    plt.plot(range(len(rewards_over_episodes)), rewards_over_episodes)
    plt.title("MAX Q Learning rewards over episodes")
    plt.ylabel("Maximum Reward")
    plt.xlabel("Episodes")
    plt.show()

    policy = np.ones((len(max_q_learner.q_values), len(max_q_learner.actions)))

    for i, state in enumerate(max_q_learner.q_values):
        policy[i] = max_q_learner.q_values[state]
    plt.imshow(policy, aspect='auto')
    plt.colorbar()
    plt.show()

    plot_title = f'Max Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    # grid = np.array([list(e.values()) for e in max_q_learner.state_visit]).T
    # # plt.imshow(grid, aspect='auto')
    # fig1, ax = plt.subplots(1, sharex=True, sharey=False)
    # ax.imshow(grid, interpolation='none', aspect='auto')
    # # ax2.imshow(grid, interpolation='bicubic', aspect='auto')
    # for (j, i), label in np.ndenumerate(grid):
    #     ax.text(i, j, label, ha='center', va='center')
    #     # ax2.text(i, j, label, ha='center', va='center')
    # # plt.colorbar()
    # plt.show()

    logger.info(f"Removed edges: {edges}")
    # max_q_learner.save_model(f"models/mql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.now()}_.pkl")
