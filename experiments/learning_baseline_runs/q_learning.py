from ptnrue.baselines.rl.expected_q_learning_basleline import ExpectedQLearner
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    EgalitarianTheilReward,
)
import logging
from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
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
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=15)
    episodes = 150

    q_learner = ExpectedQLearner(base_graph=g, reward=reward, edge_types=edge_types,
                                 budget=budget, episodes=episodes, step_size=1)
    rewards_over_episodes = q_learner.train(return_rewards_over_episodes=True)
    rewards, edges = q_learner.inference()

    plt.plot(range(len(rewards_over_episodes)), rewards_over_episodes)
    plt.title("Q Learning rewards over episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward (Return)")
    plt.show()

    policy = np.ones((len(q_learner.q_values), len(q_learner.actions)))

    for i, state in enumerate(q_learner.q_values):
        policy[i] = q_learner.q_values[state]
    # extent=[xmin,xmax,ymin,ymax]
    plt.imshow(policy, aspect='auto', extent=[0,len(q_learner.actions),0,len(q_learner.q_values)])
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.colorbar()
    plt.show()

    plot_title = f'Q Learning solution with {reward.__class__.__name__} and budget size {budget}'
    fig, ax = plot_rewards_and_graphs(g, [(rewards, edges)], plot_title)
    plt.show()

    logger.info(f"Removed edges: {edges}")
    # q_learner.save_model(f"models/ql_{episodes}_{reward.__class__.__name__}_{budget}_{datetime.datetime.now()}_.pkl")
