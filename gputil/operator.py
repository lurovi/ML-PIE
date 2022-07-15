from abc import ABC, abstractmethod
from gputil.tree import *


# ==============================================================================================================
# SELECTION
# ==============================================================================================================


class Selection(ABC):

    @abstractmethod
    def select(self, population: List[PrimitiveTree]) -> List[PrimitiveTree]:
        pass


# ==============================================================================================================
# CROSSOVER
# ==============================================================================================================


class Crossover(ABC):

    @abstractmethod
    def cross(self, trees: List[PrimitiveTree]) -> List[PrimitiveTree]:
        pass


# ==============================================================================================================
# MUTATION
# ==============================================================================================================


class Mutation(ABC):

    @abstractmethod
    def mute(self, tree: PrimitiveTree) -> PrimitiveTree:
        pass


class ShrinkMutation(Mutation):

    def mute(self, tree: PrimitiveTree) -> PrimitiveTree:
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


class UniformMutation(Mutation):

    def mute(self, tree: PrimitiveTree) -> PrimitiveTree:
        layer_ind_mut = random.randint(0, tree.depth()-1)
        node_ind_mut = random.randint(0, tree.number_of_nodes_at_layer(layer_ind_mut)-1)
        node_type = tree.get_node_type(tree.node(layer_ind_mut, node_ind_mut))
        if (layer_ind_mut == tree.depth() - 1) or (tree.is_leaf(layer_ind_mut, node_ind_mut) and not(tree.primitive_set().is_there_type(node_type))):
            return tree.replace_subtree(
                PrimitiveTree(
                    gen_simple_leaf_tree_as_list(
                        tree.terminal_set().sample_typed(node_type), tree.max_degree(), tree.max_depth() - layer_ind_mut
                    ),
                    tree.primitive_set(), tree.terminal_set()
                ),
                layer_ind_mut, node_ind_mut
            )
        new_pset = tree.primitive_set().change_return_type(node_type)
        return tree.replace_subtree(
            gen_half_half(new_pset, tree.terminal_set(), 1, tree.max_depth() - layer_ind_mut),
            layer_ind_mut, node_ind_mut
        )
