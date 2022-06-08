from experiments.baselines.rl.max_q_learner import MaxQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from experiments.rewards import (
    egalitarian_theil,
    utilitarian,
    simple,
)
import logging
from experiments.plotting.solution_plotting import plot_rewards_and_graphs
from matplotlib import pyplot as plt
from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    g = ig.load("./base_data/graph_1.gml")
    census_data = gpd.read_file("./base_data/census_data_1.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    budget = 9
    reward_func = simple
    epochs = 1000

    max_q_learner = MaxQLearner(g, reward_func, census_data, edge_types, budget, epochs)
    rewards_over_epochs = max_q_learner.train(return_rewards_over_epochs=True)
    rewards, edges = max_q_learner.inference()

    plt.plot(range(len(rewards_over_epochs)), rewards_over_epochs)
    plt.title("MAX Q Learning rewards over epochs")
    plt.ylabel("Maximum Reward")
    plt.xlabel("Episodes")
    plt.show()

    policy = np.ones((len(max_q_learner.q_values), len(max_q_learner.actions)))

    for i, state in enumerate(max_q_learner.q_values):
        policy[i] = max_q_learner.q_values[state]
    plt.imshow(policy, aspect='auto')
    plt.colorbar()
    plt.show()

    plot_title = f'Max Q Learning solution with {reward_func.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    logger.info(f"Removed edges: {edges}")
    # max_q_learner.save_model(f"models/mql_{epochs}_{reward_func.__name__}_{budget}_{datetime.now()}_.pkl")
