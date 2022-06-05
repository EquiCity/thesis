from typing import List, Tuple
import igraph as ig
import pandas as pd
import pygad
import logging
from ..rewards.egalitarian import egalitarian_theil


logger = logging.getLogger(__name__)


def ga_baseline(g: ig.Graph, census_data: pd.DataFrame, edge_types: List[str],
                budget: int = 5, reward_func: callable = egalitarian_theil, num_generations: int = 200,
                num_parents_mating: int = 5, sol_per_pop: int = 5, crossover_probability: float = 0.4,
                mutation_probability: float = 0.4, saturation: int = 20) -> Tuple[List[float], List[ig.Edge]]:
    """

    Args:
        g:
        census_data:
        edge_types:
        budget:
        reward_func:
        num_generations:
        num_parents_mating:
        sol_per_pop:
        crossover_probability:
        mutation_probability:
        saturation:

    Returns:

    """
    removable_edges = g.es.select(type_in=edge_types).indices

    def individual_fitness(solution: List[int], solution_idx: int):
        edges_to_delete = [(e.source_vertex.index, e.target_vertex.index) for e in g.es[solution]]
        g_prime = g.copy()
        g_prime.delete_edges(edges_to_delete)
        r = reward_func(g_prime, census_data)
        return r

    ga_instance = pygad.GA(
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=individual_fitness,
        initial_population=None,
        sol_per_pop=sol_per_pop,
        num_genes=budget,
        gene_type=int,
        parent_selection_type="sss",
        crossover_type="single_point",
        crossover_probability=crossover_probability,
        mutation_type="random",
        mutation_probability=mutation_probability,
        mutation_by_replacement=False,
        mutation_percent_genes="default",
        mutation_num_genes=None,
        gene_space=list(removable_edges),
        # on_start=None,
        # on_fitness=None,
        # on_parents=None,
        # on_crossover=None,
        # on_mutation=None,
        # callback_generation=None,
        # on_generation=None,
        # on_stop=None,
        delay_after_gen=0.0,
        save_best_solutions=True,
        save_solutions=False,
        suppress_warnings=False,
        stop_criteria=f'saturate_{saturation}',
    )

    logger.info("Starting GA run")
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    logger.info("Parameters of the best solution : {solution}".format(solution=solution))
    logger.info("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    return solution_fitness, solution
