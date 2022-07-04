import pickle

from ptnrue.baselines import (
    optimal_baseline,
    random_baseline,
    greedy_baseline,
    ga_baseline,
    ExpectedQLearner,
    MaxQLearner,
)
import igraph as ig
import numpy as np
import geopandas as gpd
from ptnrue.rewards import (
    CustomReward,
)
import logging
from matplotlib import pyplot as plt
from ptnrue.plotting.solution_plotting import plot_rewards_and_graphs
from ptnrue.constants.travel_metric import TravelMetric

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    dataset = 5
    g: ig.Graph = ig.load(f"../base_data/graph_{dataset}.gml")
    census_data = gpd.read_file(f"../base_data/census_data_{dataset}.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    num_edges = len(g.es.select(type_in=edge_types))
    budget = 3

    com_threshold = 15
    considered_metrics = [TravelMetric.TT, TravelMetric.HOPS, TravelMetric.COM]

    reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))

    reward = CustomReward(reward_dict=reward_dict, census_data=census_data,
                          com_threshold=com_threshold)
    # Random Baseline
    solution_random = random_baseline(g=g, reward=reward,
                                     edge_types=edge_types, budget=budget)

    # Optimal Baseline
    solutions_optimal = optimal_baseline(g=g, edge_types=edge_types,
                                         budget=budget, reward=reward)
    solution_optimal = solutions_optimal[0]

    # Greedy Baseline
    solution_greedy = greedy_baseline(g=g, reward=reward, edge_types=edge_types, budget=budget)

    # # GA Baseline
    solution_ga = ga_baseline(g=g, reward=reward, edge_types=edge_types,
                                budget=budget, num_generations=50, sol_per_pop=20,
                                num_parents_mating=10, saturation=20,
                                mutation_probability=0.5, crossover_probability=0.5)

    # # Q-Learning Baseline
    episodes = 30
    q_learner = ExpectedQLearner(base_graph=g, reward=reward, edge_types=edge_types,
                                 budget=budget, episodes=episodes, step_size=1)
    rewards_over_episodes_q = q_learner.train(return_rewards_over_episodes=True)
    solution_q = q_learner.inference()

    # Max-Q-Learning Baseline
    max_q_learner = MaxQLearner(g, reward, edge_types, budget, episodes, step_size=1)
    rewards_over_episodes_maxq = max_q_learner.train(return_rewards_over_episodes=True)
    max_q_learner.reset_state_visits()
    solution_maxq = max_q_learner.inference()

    all_solutions = [
        solution_random,
        solution_optimal,
        solution_greedy,
        solution_ga,
        solution_q,
        solution_maxq,
    ]

    plot_rewards_and_graphs(g, all_solutions, "Random, Opt, Greedy, GA, Q, MaxQ",
                            yticks=list(np.arange(20,65,5)))
    plt.savefig("./plots/synth_data_comparison_all_methods.svg")
    plt.show()
