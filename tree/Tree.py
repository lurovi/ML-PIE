from abc import ABC, abstractmethod


class Tree(ABC):
    def __init__(self, max_degree, max_depth, domain):
        self.__max_degree = max_degree
        self.__max_depth = max_depth
        self.__domain = domain

        self.__max_number_of_nodes = 0
        for i in range(self.__max_depth):
            if i == self.__max_depth - 1:
                self.__max_breadth = self.__max_degree ** i
            self.__max_number_of_nodes += self.__max_degree ** i

    def max_degree(self):
        return self.__max_degree

    def max_depth(self):
        return self.__max_depth

    def max_number_of_nodes(self):
        return self.__max_number_of_nodes

    def max_breadth(self):
        return self.__max_breadth

    def domain(self):
        return self.__domain

    @abstractmethod
    def layer(self, layer_ind):
        pass

    @abstractmethod
    def root(self):
        pass

    @abstractmethod
    def number_of_nodes(self):
        pass

    @abstractmethod
    def number_of_nodes_at_layer(self, layer_ind):
        pass

    @abstractmethod
    def depth(self):
        pass

    @abstractmethod
    def actual_max_breadth(self):
        pass

    @abstractmethod
    def actual_max_degree(self):
        pass

    @abstractmethod
    def layer_of_max_breadth(self):
        pass

    @abstractmethod
    def leaf_nodes(self):
        pass

    @abstractmethod
    def internal_nodes(self):
        pass

    @abstractmethod
    def node(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def is_leaf(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def siblings(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def children(self, layer_ind, node_ind):
        pass

    @abstractmethod
    def parent(self, layer_ind, node_ind):
        pass
