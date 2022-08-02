from typing import List
from gp.tree.PrimitiveTree import PrimitiveTree
from gp.operator.Crossover import Crossover
import random


class OnePointCrossover(Crossover):

    def mate(self, individuals: List[PrimitiveTree]) -> List[PrimitiveTree]:
        tree_1, tree_2 = individuals[0], individuals[1]
        if tree_1.depth() > tree_2.depth():
            tree_1, tree_2 = tree_2, tree_1
        iter_1 = tree_1.node_indexes_iterate()
        iter_2 = tree_2.node_indexes_iterate()
        candidates = []
        for layer_ind_1, node_ind_1 in iter_1:
            node_type_1 = tree_1.get_node_type(tree_1.node(layer_ind_1, node_ind_1))
            for layer_ind_2, node_ind_2 in iter_2:
                node_type_2 = tree_2.get_node_type(tree_2.node(layer_ind_2, node_ind_2))
                if node_type_1 == node_type_2 and (tree_2.subtree_depth(layer_ind_2, node_ind_2) <= tree_1.max_depth() - layer_ind_1) and (tree_1.subtree_depth(layer_ind_1, node_ind_1) <= tree_2.max_depth() - layer_ind_2):
                    candidates.append([(layer_ind_1, node_ind_1), (layer_ind_2, node_ind_2)])
        if not candidates:
            return [tree_1.copy(), tree_2.copy()]
        candidate = random.choice(candidates)
        new_tree_1 = tree_1.extract_subtree(candidate[0][0], candidate[0][1])
        new_tree_2 = tree_2.extract_subtree(candidate[1][0], candidate[1][1])
        return [tree_1.replace_subtree(new_tree_2, candidate[0][0], candidate[0][1]), tree_2.replace_subtree(new_tree_1, candidate[1][0], candidate[1][1])]
