import random
from gp.tree.PrimitiveTree import PrimitiveTree
from gp.operator.Mutation import Mutation


class ShrinkMutation(Mutation):

    def mutate(self, individual: PrimitiveTree) -> PrimitiveTree:
        tree = individual
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
            return tree.copy()
        candidate = random.choice(candidates)
        new_tree = tree.extract_subtree(candidate[1][0], candidate[1][1])
        return tree.replace_subtree(new_tree, candidate[0][0], candidate[0][1])
