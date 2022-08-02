from gp.operator.Mutation import Mutation
import random
from gp.tree.PrimitiveTree import PrimitiveTree
from gp.tree.HalfHalfGenerator import HalfHalfGenerator

from gp.tree.TreeGenerator import TreeGenerator


class UniformMutation(Mutation):

    def mutate(self, individual: PrimitiveTree) -> PrimitiveTree:
        tree = individual
        layer_ind_mut = random.randint(0, tree.depth()-1)
        node_ind_mut = random.randint(0, tree.number_of_nodes_at_layer(layer_ind_mut)-1)
        node_type = tree.get_node_type(tree.node(layer_ind_mut, node_ind_mut))
        if (layer_ind_mut == tree.max_depth() - 1) or (tree.is_leaf(layer_ind_mut, node_ind_mut) and not(tree.primitive_set().is_there_type(node_type))):
            return tree.replace_subtree(
                PrimitiveTree(
                    TreeGenerator.gen_simple_leaf_tree_as_list(
                        tree.terminal_set().sample_typed(node_type), tree.max_degree(), tree.max_depth() - layer_ind_mut
                    ),
                    tree.primitive_set(), tree.terminal_set()
                ),
                layer_ind_mut, node_ind_mut
            )
        new_pset = tree.primitive_set().change_return_type(node_type)
        return tree.replace_subtree(
            HalfHalfGenerator(new_pset, tree.terminal_set(), 1, tree.max_depth() - layer_ind_mut).generate_tree(),
            layer_ind_mut, node_ind_mut
        )
