from typing import Tuple, List

from gp.evolution.Population import Population
from gp.operator.Selection import Selection
from gp.tree.PrimitiveTree import PrimitiveTree


class TournamentSelection(Selection):
    def __init__(self, num_individuals: int, tournament_size: int):
        self.num_individuals = num_individuals
        self.tournament_size = tournament_size

    def select(self, population: Population) -> Tuple[List[PrimitiveTree], List[List[float]]]:
        new_population = []
        new_fitness_eval = []
        n_sample = list(range(population.get_size()))
        for _ in range(self.num_individuals):
            tourn_indexes = random.choices(n_sample, k=self.tournament_size)
            tourn_population = []
            tourn_fitness_eval = []
            for i in tourn_indexes:
                tourn_population.append(population.get_individual(i))
                tourn_fitness_eval.append(population.get_fitness(i))
            sort_ind = [(iii, tourn_fitness_eval[iii]) for iii in range(len(tourn_fitness_eval))]
            sort_ind.sort(key=lambda x: x[1])
            best_ind = sort_ind[0][0]
            new_population.append(tourn_population[best_ind])
            new_fitness_eval.append(tourn_fitness_eval[best_ind])
        return new_population, new_fitness_eval
