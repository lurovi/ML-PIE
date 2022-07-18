import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from gputil.evolution import Population, Individual, PrimitiveTreeIndividual
from gputil.tree import *


# ==============================================================================================================
# SELECTION
# ==============================================================================================================


class Selection(ABC):

    @abstractmethod
    def select(self, population: Population) -> Tuple[List[Individual], List[List[float]]]:
        pass


class TournamentSelection(Selection):
    def __init__(self, num_individuals: int, tournament_size: int):
        self.num_individuals = num_individuals
        self.tournament_size = tournament_size

    def select(self, population: Population) -> Tuple[List[Individual], List[List[float]]]:
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


# ==============================================================================================================
# CROSSOVER
# ==============================================================================================================


class Crossover(ABC):

    @abstractmethod
    def cross(self, individuals: List[Individual]) -> List[Individual]:
        pass


class OnePointCrossover(Crossover):

    def cross(self, individuals: List[Individual]) -> List[Individual]:
        tree_1, tree_2 = individuals[0].get_individual(), individuals[1].get_individual()
        if tree_1.depth() > tree_2.depth():
            tree_1, tree_2 = tree_2, tree_1
        iter_1 = tree_1.node_indexes_iterate()
        candidates = []
        for layer_ind_1, node_ind_1 in iter_1:
            node_type_1 = tree_1.get_node_type(tree_1.node(layer_ind_1, node_ind_1))
            iter_2 = tree_2.node_indexes_iterate()
            for layer_ind_2, node_ind_2 in iter_2:
                node_type_2 = tree_2.get_node_type(tree_2.node(layer_ind_2, node_ind_2))
                if node_type_1 == node_type_2 and (tree_2.subtree_depth(layer_ind_2, node_ind_2) <= tree_1.max_depth() - layer_ind_1) and (tree_1.subtree_depth(layer_ind_1, node_ind_1) <= tree_2.max_depth() - layer_ind_2):
                    candidates.append([(layer_ind_1, node_ind_1), (layer_ind_2, node_ind_2)])
        if not candidates:
            return [tree_1.copy(), tree_2.copy()]
        candidate = random.choice(candidates)
        new_tree_1 = tree_1.extract_subtree(candidate[0][0], candidate[0][1])
        new_tree_2 = tree_2.extract_subtree(candidate[1][0], candidate[1][1])
        return [PrimitiveTreeIndividual(tree_1.replace_subtree(new_tree_2, candidate[0][0], candidate[0][1])), PrimitiveTreeIndividual(tree_2.replace_subtree(new_tree_1, candidate[1][0], candidate[1][1]))]


# ==============================================================================================================
# MUTATION
# ==============================================================================================================


class Mutation(ABC):

    @abstractmethod
    def mute(self, individual: Individual) -> Individual:
        pass


class ShrinkMutation(Mutation):

    def mute(self, individual: Individual) -> Individual:
        tree = individual.get_individual()
        candidates = []
        for layer_ind in range(tree.depth()-1):
            for node_ind in range(tree.number_of_nodes_at_layer(layer_ind)):
                if not(tree.is_leaf(layer_ind, node_ind)):
                    father_layer, father_ind = layer_ind, node_ind
                    child = tree.children(father_layer, father_ind)
                    for child_layer, child_ind, c in child:
                        if tree.get_node_type(c) == tree.get_node_type(tree.node(father_layer, father_ind)):
                            candidates.append([(father_layer, father_ind), (child_layer, child_ind)])
        if not candidates:
            return PrimitiveTreeIndividual(tree.copy())
        candidate = random.choice(candidates)
        new_tree = tree.extract_subtree(candidate[1][0], candidate[1][1])
        return PrimitiveTreeIndividual(tree.replace_subtree(new_tree, candidate[0][0], candidate[0][1]))


class UniformMutation(Mutation):

    def mute(self, individual: Individual) -> Individual:
        tree = individual.get_individual()
        layer_ind_mut = random.randint(0, tree.depth()-1)
        node_ind_mut = random.randint(0, tree.number_of_nodes_at_layer(layer_ind_mut)-1)
        node_type = tree.get_node_type(tree.node(layer_ind_mut, node_ind_mut))
        if (layer_ind_mut == tree.max_depth() - 1) or (tree.is_leaf(layer_ind_mut, node_ind_mut) and not(tree.primitive_set().is_there_type(node_type))):
            return PrimitiveTreeIndividual(tree.replace_subtree(
                PrimitiveTree(
                    gen_simple_leaf_tree_as_list(
                        tree.terminal_set().sample_typed(node_type), tree.max_degree(), tree.max_depth() - layer_ind_mut
                    ),
                    tree.primitive_set(), tree.terminal_set()
                ),
                layer_ind_mut, node_ind_mut
            ))
        new_pset = tree.primitive_set().change_return_type(node_type)
        return PrimitiveTreeIndividual(tree.replace_subtree(
            gen_half_half(new_pset, tree.terminal_set(), 1, tree.max_depth() - layer_ind_mut),
            layer_ind_mut, node_ind_mut
        ))
