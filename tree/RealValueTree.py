from tree.Tree import Tree
import numpy as np


class RealValueTree(Tree):
    def __init__(self, max_degree, max_depth, domain, data):
        super().__init__(max_degree, max_depth, domain)
        self.__tree = data  # {0: ndarray([0.50]), 1: ndarray([0.20, 0.50, 0.0]), ... }

    def __check_layer_index_with_max_depth(self, layer_ind):
        if not(0 <= layer_ind < self.max_depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def __check_layer_index_with_actual_depth(self, layer_ind):
        if not(0 <= layer_ind < self.depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def layer(self, layer_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        return self.__tree[layer_ind]

    def root(self):
        return self.__tree[0][0]

    def number_of_nodes(self):
        n_nodes = 0
        for i in range(self.max_depth()):
            nodes = self.layer(i)
            n_nodes += (nodes != 0).sum()
        return n_nodes

    def number_of_nodes_at_layer(self, layer_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        return (self.layer(layer_ind) != 0).sum()

    def depth(self):
        for i in range(self.max_depth()):
            if np.all( self.layer(i) == 0):
                return i
        return self.max_depth()

    def actual_max_breadth(self):
        max_layer = -1000000
        for i in range(self.max_depth()):
            n_nodes = self.number_of_nodes_at_layer(i)
            if ( n_nodes > max_layer):
                max_layer = n_nodes
        return max_layer

    def actual_max_degree(self):
        max_degree = -1000000
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = np.where(curr_layer != 0)
            for j in range(len(ind)):
                if self.children(i, j).size > max_degree:
                    max_degree = self.children(i, j).size
        return max_degree

    def layer_of_max_breadth(self):
        max_layer = -1000000
        ind = -1
        for i in range(self.max_depth()):
            n_nodes = (self.layer(i) != 0).sum()
            if (n_nodes > max_layer):
                max_layer = n_nodes
                ind = i
        return ind

    def leaf_nodes(self):
        leaf_nodes = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = np.where(curr_layer != 0)
            for j in range(len(ind)):
                if self.is_leaf(i, j):
                    leaf_nodes.append(curr_layer[ind[j]])
        return np.array(leaf_nodes)

    def internal_nodes(self):
        internal_nodes = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = np.where(curr_layer != 0)
            for j in range(len(ind)):
                if not(self.is_leaf(i, j)):
                    internal_nodes.append(curr_layer[ind[j]])
        return np.array(internal_nodes)

    def node(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = np.where(self.layer(layer_ind) != 0)
        elem = curr_layer[elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return curr_node

    def is_leaf(self, layer_ind, node_ind):
        return self.children(layer_ind, node_ind).size == 0

    def siblings(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = np.where(self.layer(layer_ind) != 0)
        elem = curr_layer[elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return elem[:node_ind], curr_node, elem[(node_ind+1):]

    def children(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        if layer_ind == self.depth() - 1:
            return np.array([])
        curr_layer = self.layer(layer_ind)
        next_layer = self.layer(layer_ind + 1)
        elem_ind = np.where(curr_layer != 0)
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind = elem_ind[node_ind]
        children = []
        start_ind = self.max_degree() * ind
        for i in range(start_ind, start_ind + self.max_degree()):
            if next_layer[i] != 0:
                children.append(next_layer[i])
        return np.array(children)

    def parent(self, layer_ind, node_ind):
        self.__check_layer_index_with_actual_depth(layer_ind)
        if layer_ind == 0:
            return None
        curr_layer = self.layer(layer_ind)
        previous_layer = self.layer(layer_ind - 1)
        elem_ind = np.where(curr_layer != 0)
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind_of_node = elem_ind[node_ind]
        curr_ind = ind_of_node//self.max_degree()
        return previous_layer[curr_ind]
