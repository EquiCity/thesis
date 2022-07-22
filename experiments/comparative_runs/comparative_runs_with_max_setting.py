import pickle
import random

import pandas as pd
import torch

from eptnr.baselines import (
    optimal_max_baseline,
    random_baseline,
    greedy_baseline,
    ga_max_baseline,
    ExpectedQLearner,
    MaxQLearner,
)
from eptnr.deep_q_learning_approach import (
    DeepQLearner,
    DeepMaxQLearner,
)
from eptnr import plotting
import igraph as ig
import numpy as np
import geopandas as gpd
from eptnr.rewards import (
    CustomReward,
    EgalitarianTheilReward,
)
import logging
from matplotlib import pyplot as plt
from eptnr.plotting.solution_plotting import plot_rewards_and_graphs
import torch
import random

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    dataset = 3
    g: ig.Graph = ig.load(f"../base_data/graph_{dataset}.gml")
    census_data = gpd.read_file(f"../base_data/census_data_{dataset}.geojson")
    edge_types = list(np.unique(g.es['type']))
    edge_types.remove('walk')
    num_edges = len(g.es.select(type_in=edge_types))
    budget = 9
    com_threshold = 15

    # reward_dict = pickle.load(open(f"../base_data/reward_dict_{dataset}.pkl", 'rb'))
    # reward = CustomReward(reward_dict=reward_dict, census_data=census_data,
    #                       com_threshold=com_threshold)
    reward = EgalitarianTheilReward(census_data=census_data, com_threshold=com_threshold)

    runs = 16
    random_seeds = np.arange(0, runs) * 2048

    results = {
        'random': [],
        'optimal': [],
        'greedy': [],
        'ga': [],
        'q learning': [],
        'maxq learning': [],
        'DQN learning': [],
        'DMaxQN learning': [],
    }

    # GA
    num_generations = 150

    # Deep learning setup
    episodes = 150
    batch_size = 32
    replay_memory_size = 512
    target_network_update_step = 50

    # EPS Schedule
    eps_start = 1.0
    eps_end = 0.001
    eps_decay = 500
    static_eps_steps = 100 * budget

    for rs in random_seeds:
        # Set random seed
        np.random.seed(rs)
        random.seed(rs)
        torch.manual_seed(rs)

        # Random Baseline
        solution_random = random_baseline(g=g, reward=reward,
                                          edge_types=edge_types, budget=budget)

        # Optimal Baseline
        solutions_optimal = optimal_max_baseline(g=g, edge_types=edge_types,
                                                 budget=budget, reward=reward)
        solution_optimal = solutions_optimal[0]

        # Greedy Baseline
        solution_greedy = greedy_baseline(g=g, reward=reward, edge_types=edge_types, budget=budget)

        # GA Baseline
        solution_ga = ga_max_baseline(g=g, reward=reward, edge_types=edge_types,
                                      budget=budget, num_generations=num_generations, sol_per_pop=20,
                                      num_parents_mating=10, saturation=20,
                                      mutation_probability=0.5, crossover_probability=0.5)

        # Q-Learning Baseline
        q_learner = ExpectedQLearner(base_graph=g, reward=reward, edge_types=edge_types,
                                     budget=budget, episodes=episodes, step_size=1)
        rewards_over_episodes_q = q_learner.train(return_rewards_over_episodes=True)
        solution_q = q_learner.inference()

        # Max-Q-Learning Baseline
        max_q_learner = MaxQLearner(g, reward, edge_types, budget, episodes, step_size=1)
        rewards_over_episodes_maxq = max_q_learner.train(return_rewards_over_episodes=True)
        max_q_learner.reset_state_visits()
        solution_maxq = max_q_learner.inference()

        # DQN
        dqn_learner = DeepQLearner(g, reward, edge_types, budget, episodes,
                                   batch_size=batch_size, replay_memory_size=replay_memory_size,
                                   target_network_update_step=target_network_update_step)
        rewards_over_episodes_dqn = dqn_learner.train(return_max_rewards_over_episodes=True,
                                                      return_cum_rewards_over_episodes=False,
                                                      return_epsilon_over_episodes=False)
        dqn_learner.reset_state_visits()
        solution_dqn = dqn_learner.inference()

        # DMaxQN
        dmaxqn_learner = DeepMaxQLearner(g, reward, edge_types, budget, episodes,
                                         batch_size=batch_size, replay_memory_size=replay_memory_size,
                                         target_network_update_step=target_network_update_step)
        rewards_over_episodes_dmaxqn = dmaxqn_learner.train(return_max_rewards_over_episodes=True,
                                                            return_cum_rewards_over_episodes=False,
                                                            return_epsilon_over_episodes=False)
        dmaxqn_learner.reset_state_visits()
        solution_dmaxqn = dmaxqn_learner.inference()

        all_solutions = [
            solution_random,
            solution_optimal,
            solution_greedy,
            solution_ga,
            solution_q,
            solution_maxq,
            solution_dqn,
            solution_dmaxqn,
        ]

        # Append the maximum reward value to the results
        results['random'].append(max(solution_random[0]))
        results['optimal'].append(max(solution_optimal[0]))
        results['greedy'].append(max(solution_greedy[0]))
        results['ga'].append(max(solution_ga[0]))
        results['q learning'].append(max(solution_q[0]))
        results['maxq learning'].append(max(solution_maxq[0]))
        results['DQN learning'].append(max(solution_dqn[0]))
        results['DMaxQN learning'].append(max(solution_dmaxqn[0]))

        if runs == 1:
            plot_rewards_and_graphs(g, all_solutions, "Random, Opt Max, Greedy, GA Max, Q, MaxQ",
                                    yticks=list(np.arange(0, 110, 10)))
            # plt.savefig("./plots/synth_data_comparison_all_MAX_methods.svg")
            plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))

    for res in results:
        ax.plot(random_seeds, results[res], label=res, alpha=0.5, linestyle='-' if 'max' in res.lower() else '--')
        results[res] = results[res][:-1]

    ax.set_xticks(random_seeds[::5])
    ax.set_ylabel('Maximum reward')
    ax.set_xlabel('Random seed')
    ax.legend()
    fig.tight_layout()
    plt.show()

    print(pd.DataFrame(results).describe().to_latex())

    # fig.savefig(f'./plots/synth_ds_{dataset}_{budget}_comparison_all_MAX_methods_over_random_seeds.svg')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/figures/synth_ds_{dataset}_{budget}_comparison_all_MAX_methods_over_random_seeds.png')
    # fig.savefig(
    #     f'/home/rico/Documents/thesis/paper/overleaf/62a466789b2183065a639cda/content-media/synth_ds_{dataset}_{budget}_comparison_all_MAX_methods_over_random_seeds.png')
