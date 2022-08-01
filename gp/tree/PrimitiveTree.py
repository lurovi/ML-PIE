from typing import List

from gp.tree.PrimitiveSet import PrimitiveSet
from gp.tree.TerminalSet import TerminalSet


class PrimitiveTree:
    def __init__(self, data: List[List[str]], primitive_set: PrimitiveSet, terminal_set: TerminalSet):
        self.__max_degree = primitive_set.max_arity()
        self.__max_depth = len(data)

        self.__max_number_of_nodes = 0
        for i in range(self.__max_depth):
            if i == self.__max_depth - 1:
                self.__max_breadth = self.__max_degree ** i
            self.__max_number_of_nodes += self.__max_degree ** i

        #    [ ["+"],
        #   ["+", "*"],
        #  ["-", "x3", "c2 20", "^2"],
        # ["x0", "x5", "", "", "", "", "e0 0.24535563", ""] ]
        self.__tree = data
        self.__primitive_set = primitive_set
        self.__terminal_set = terminal_set
        self.__type_dict = {}
        for name in self.__primitive_set.primitive_names():
            self.__type_dict[name] = self.__primitive_set.get_primitive(name).return_type()
        for i in range(len(self.__terminal_set.all_idx())):
            self.__type_dict[self.__terminal_set.all_idx()[i]] = self.__terminal_set.get_type(i)

    def copy(self):
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        return PrimitiveTree(tre, self.__primitive_set, self.__terminal_set)

    def get_node_string_without_value(self, node_string):
        if self.__primitive_set.is_primitive(node_string):
            return node_string
        elif node_string[0] == "x":
            return node_string
        else:
            return node_string[:node_string.find(" ")]

    def get_node_type(self, node_string):
        return self.__type_dict[self.get_node_string_without_value(node_string)]

    def max_degree(self):
        return self.__max_degree

    def max_depth(self):
        return self.__max_depth

    def max_number_of_nodes(self):
        return self.__max_number_of_nodes

    def max_breadth(self):
        return self.__max_breadth

    def primitive_set(self):
        return self.__primitive_set

    def terminal_set(self):
        return self.__terminal_set

    def __len__(self):
        return self.number_of_nodes()

    def print_as_text(self):
        return self.__str__()

    def __str__(self):
        s = "  "
        stack = [(0, 0, self.layer(0)[0])]
        while len(stack) > 0:
            val = stack.pop()
            if type(val) == str:
                s += val + "  "
            else:
                curr_i, curr_j, curr_val = val
                children = self.children(curr_i, curr_j)
                s += curr_val + "  "
                if len(children) != 0:
                    stack.append(")")
                    for iii in reversed(range(len(children))):
                        stack.append(children[iii])
                    stack.append("(")
        return s

    def print_as_tree(self):
        s = ""
        n_indent = self.depth()-1
        s += "\n"
        for i in range(self.depth()):
            for _ in range(n_indent):
                s += " "
            n_indent -= 1
            curr_layer = self.layer(i)
            curr_elem = ["["+elem+"]\t" for elem in curr_layer if elem != ""]
            for elem in curr_elem:
                s += elem
            s += "\n"
        return s

    def __check_layer_index_with_max_depth(self, layer_ind: int):
        if not(0 <= layer_ind < self.max_depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def __check_layer_index_with_actual_depth(self, layer_ind: int):
        if not(0 <= layer_ind < self.depth()):
            raise IndexError(f"{layer_ind} is out of range as layer index.")

    def layer(self, layer_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        return self.__tree[layer_ind]

    def root(self):
        return self.__tree[0][0]

    def flatten(self):
        tre = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            tre.append([n for n in curr_layer if n != ""])
        return tre

    def number_of_nodes(self):
        n_nodes = 0
        for i in range(self.max_depth()):
            nodes = self.layer(i)
            n_nodes += sum([1 if n != "" else 0 for n in nodes])
        return n_nodes

    def count_primitives(self):
        dic = {}
        for p in self.primitive_set().primitive_names():
            dic[p] = 0.0
            for p0 in self.primitive_set().primitive_names():
                lp = sorted([p, p0])
                dic[(lp[0], lp[1])] = 0.0

        for layer_ind in range(self.depth()):
            curr_layer = self.layer(layer_ind)
            elem = [curr_layer[i] for i in range(len(curr_layer)) if curr_layer[i] != ""]
            prim = []
            for i in range(len(elem)):
                if self.primitive_set().is_primitive(elem[i]):
                    prim.append((elem[i], [child[2] for child in self.children(layer_ind, i) if self.primitive_set().is_primitive(child[2])]))
            for pr, child in prim:
                dic[pr] += 1.0
                for c in child:
                    lp = sorted([pr, c])
                    dic[(lp[0], lp[1])] += 1.0
        return dic

    def number_of_nodes_at_layer(self, layer_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        return sum([1 if n != "" else 0 for n in self.layer(layer_ind)])

    def depth(self):
        for i in range(self.max_depth()):
            if all([n == "" for n in self.__tree[i]]):
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
            ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
            for j in range(len(ind)):
                if len(self.children(i, j)) > max_degree:
                    max_degree = len(self.children(i, j))
        return max_degree

    def layer_of_max_breadth(self):
        max_layer = -1000000
        ind = -1
        for i in range(self.max_depth()):
            n_nodes = self.number_of_nodes_at_layer(i)
            if (n_nodes > max_layer):
                max_layer = n_nodes
                ind = i
        return ind

    def leaf_nodes(self):
        leaf_nodes = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
            for j in range(len(ind)):
                if self.is_leaf(i, j):
                    leaf_nodes.append(curr_layer[ind[j]])
        return leaf_nodes

    def internal_nodes(self):
        internal_nodes = []
        for i in range(self.depth()):
            curr_layer = self.layer(i)
            ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
            for j in range(len(ind)):
                if not(self.is_leaf(i, j)):
                    internal_nodes.append(curr_layer[ind[j]])
        return internal_nodes

    def node(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        elem = [curr_layer[iii] for iii in elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return curr_node

    def is_leaf(self, layer_ind: int, node_ind: int):
        return len(self.children(layer_ind, node_ind)) == 0

    def siblings(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        elem = [curr_layer[iii] for iii in elem_ind]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        curr_node = elem[node_ind]
        return elem[:node_ind], curr_node, elem[(node_ind+1):]

    def children(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        if layer_ind == self.depth() - 1:
            return []
        curr_layer = self.layer(layer_ind)
        next_layer = self.layer(layer_ind + 1)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        next_elem_ind = [iii for iii in range(len(next_layer)) if next_layer[iii] != ""]
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind = elem_ind[node_ind]
        children = []
        start_ind = self.max_degree() * ind
        relative_ind = -1
        for i in range(start_ind, start_ind + self.max_degree()):
            if next_layer[i] != "":
                for iii in range(len(next_elem_ind)):
                    if next_elem_ind[iii] == i:
                        relative_ind = iii
                        break
                children.append((layer_ind + 1, relative_ind, next_layer[i]))
        return children

    def parent(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_max_depth(layer_ind)
        if layer_ind == 0:
            return None
        curr_layer = self.layer(layer_ind)
        previous_layer = self.layer(layer_ind - 1)
        elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        previous_elem_ind = [iii for iii in range(len(previous_layer)) if previous_layer[iii] != ""]
        if not (0 <= node_ind < len(elem_ind)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        ind_of_node = elem_ind[node_ind]
        curr_ind = ind_of_node//self.max_degree()
        relative_ind = -1
        for iii in range(len(previous_elem_ind)):
            if previous_elem_ind[iii] == curr_ind:
                relative_ind = iii
                break
        return (layer_ind - 1, relative_ind, previous_layer[curr_ind])

    def node_indexes_iterate(self):
        candidates = []
        for layer_ind in range(self.depth()):
            for node_ind in range(self.number_of_nodes_at_layer(layer_ind)):
                candidates.append((layer_ind, node_ind))
        return candidates

    def extract_subtree(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_actual_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        tre = [[curr_layer[elem[node_ind]]]]
        first_previous_node_abs_index = elem[node_ind]
        curr_layer_ind = layer_ind + 1
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree()**(curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            tre.append(self.layer(curr_layer_ind)[start_ind:start_ind + curr_dim])
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
        for i in range(self.max_depth() - layer_ind - 1, self.max_depth()):
            curr_dim = self.max_degree() ** (curr_layer_ind - layer_ind)
            tre.append([""]*curr_dim)
            curr_layer_ind += 1
        return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())

    def subtree_depth(self, layer_ind: int, node_ind: int) -> int:
        self.__check_layer_index_with_actual_depth(layer_ind)
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        first_previous_node_abs_index = elem[node_ind]
        depth = 1
        curr_layer_ind = layer_ind + 1
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree() ** (curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            lll = self.layer(curr_layer_ind)[start_ind:start_ind + curr_dim]
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
            lll = [nnn for nnn in lll if nnn != ""]
            if not(lll):
                return depth
            depth += 1
        return depth

    '''
    def remove_subtree(self, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_actual_depth(layer_ind)
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        first_previous_node_abs_index = elem[node_ind]
        tre[layer_ind][elem[node_ind]] = ""
        curr_layer_ind = layer_ind + 1
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree() ** (curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            tre[curr_layer_ind] = tre[curr_layer_ind][:start_ind] + [""]*curr_dim + tre[curr_layer_ind][start_ind+curr_dim:]
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
        return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())
    '''

    def replace_subtree(self, new_tree, layer_ind: int, node_ind: int):
        self.__check_layer_index_with_actual_depth(layer_ind)
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        curr_layer = self.layer(layer_ind)
        elem = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != ""]
        if not (0 <= node_ind < len(elem)):
            raise IndexError(f"{node_ind} is out of range as node index for layer {layer_ind}.")
        first_previous_node_abs_index = elem[node_ind]
        tre[layer_ind][elem[node_ind]] = new_tree.root()
        curr_layer_ind = layer_ind + 1
        if not(layer_ind + new_tree.depth() <= self.max_depth()):
            raise AttributeError("The new tree depth must not exceed the allowed max depth of the tree.")
        if not( self.get_node_type(tre[layer_ind][elem[node_ind]]) == self.get_node_type(self.__tree[layer_ind][elem[node_ind]])):
            raise AttributeError("The new tree return type must match the return type of the replaced subtree.")
        for i in range(self.max_depth() - layer_ind - 1):
            curr_dim = self.max_degree() ** (curr_layer_ind - layer_ind)
            start_ind = first_previous_node_abs_index * self.max_degree()
            tre[curr_layer_ind] = tre[curr_layer_ind][:start_ind] + new_tree.layer(curr_layer_ind - layer_ind) + tre[curr_layer_ind][start_ind + curr_dim:]
            curr_layer_ind += 1
            first_previous_node_abs_index = start_ind
        return PrimitiveTree(tre, self.primitive_set(), self.terminal_set())

    def extract_counting_features_from_tree(self):
        counting_dic = self.count_primitives()
        number_of_nodes = float(self.number_of_nodes())
        depth = float(self.depth())
        max_breadth = float(self.actual_max_breadth())
        max_degree = float(self.actual_max_degree())
        number_of_leaf_nodes = float(len(self.leaf_nodes()))
        number_of_internal_nodes = float(len(self.internal_nodes()))
        leaf_internal_nodes_ratio = number_of_leaf_nodes / number_of_internal_nodes
        leaf_nodes_perc = number_of_leaf_nodes / number_of_nodes
        degree_breadth_ratio = max_degree / max_breadth
        depth_number_of_nodes_ratio = depth / number_of_nodes
        keys = list(counting_dic.keys())
        single_primitives = []
        couples_primitives = []
        for k in keys:
            if isinstance(k, str):
                single_primitives.append(k)
            elif isinstance(k, tuple):
                couples_primitives.append(k)
            else:
                raise ValueError(f"Invalid key {k} found in counting dictionary key set.")
        single_primitives = sorted(single_primitives)
        couples_primitives = sorted(couples_primitives)
        counts = []
        for p in single_primitives:  # + couples_primitives:
            counts.append(float(counting_dic[p]))
        return counts + [number_of_nodes, depth, max_degree, max_breadth, depth_number_of_nodes_ratio, leaf_internal_nodes_ratio]

    @staticmethod
    def extract_counting_features_from_list_of_trees(trees: List):
        lt = []
        for tree in trees:
            lt.append(tree.extract_counting_features_from_tree())
        return lt

    def find_all_sub_chains(self):
        primitives_leafs = []
        candidates = self.node_indexes_iterate()
        for layer_ind, node_ind in candidates:
            if not(self.is_leaf(layer_ind, node_ind)) and all([True if self.is_leaf(ci,cj) else False for ci, cj, cc in self.children(layer_ind, node_ind)]):
                primitives_leafs.append((layer_ind, node_ind))
        sub_chains = []
        for layer_ind, node_ind in primitives_leafs:
            curr_chain = [self.node(layer_ind, node_ind)]
            curr_layer = layer_ind
            curr_node = node_ind
            curr_parent = self.parent(curr_layer, curr_node)
            while curr_parent is not None:
                curr_chain.append(curr_parent[2])
                curr_layer = curr_parent[0]
                curr_node = curr_parent[1]
                curr_parent = self.parent(curr_layer, curr_node)
            sub_chains.append(curr_chain)
        return sub_chains

    @staticmethod
    def weight_primitives_ranking(ranking: List[List[str]], max_weight: float = 1.0):
        part = max_weight / float(len(ranking))
        weights = {}
        curr_weight = max_weight
        for i in range(len(ranking)):
            for p in ranking[i]:
                weights[p] = curr_weight
            curr_weight -= part
        return weights

    def compute_internal_nodes_weights_average(self, ranking: List[List[str]], max_weight: float = 1.0):
        weights = PrimitiveTree.weight_primitives_ranking(ranking, max_weight)
        internal_nodes = self.internal_nodes()
        s = 0.0
        for c in internal_nodes:
            s += weights[c]
        return s/float(len(internal_nodes))

    def compute_weighted_sub_chains_average(self, ranking: List[List[str]], max_weight: float = 1.0):
        weights = PrimitiveTree.weight_primitives_ranking(ranking, max_weight)
        sub_chains = self.find_all_sub_chains()
        weighted_sub_chains = 0.0
        for i in range(len(sub_chains)):
            curr_weight = 1.0
            for j in range(len(sub_chains[i])):
                curr_weight *= weights[sub_chains[i][j]]
            weighted_sub_chains += curr_weight
        return weighted_sub_chains/float(len(sub_chains))

    def compute_property_and_weights_based_interpretability_score(self,  ranking: List[List[str]], max_weight: float = 1.0):
        counting_dic = self.count_primitives()
        number_of_nodes = float(self.number_of_nodes())
        depth = float(self.depth())
        max_breadth = float(self.actual_max_breadth())
        max_degree = float(self.actual_max_degree())
        number_of_leaf_nodes = float(len(self.leaf_nodes()))
        number_of_internal_nodes = float(len(self.internal_nodes()))
        leaf_internal_nodes_ratio = number_of_leaf_nodes / number_of_internal_nodes
        leaf_nodes_perc = number_of_leaf_nodes / number_of_nodes
        degree_breadth_ratio = max_degree / max_breadth
        depth_number_of_nodes_ratio = depth / number_of_nodes
        weights_average = self.compute_internal_nodes_weights_average(ranking, max_weight)
        weighted_sub_chains_average = self.compute_weighted_sub_chains_average(ranking, max_weight)
        return weights_average + weighted_sub_chains_average + depth_number_of_nodes_ratio + degree_breadth_ratio + leaf_nodes_perc + 1.0/number_of_nodes

    def compile(self, x: List):
        tre = [[self.__tree[i][j] for j in range(len(self.__tree[i]))] for i in range(len(self.__tree))]
        for layer_ind in reversed(range(self.depth()-1)):
            curr_layer = tre[layer_ind]
            next_layer = tre[layer_ind + 1]
            elem_ind = [iii for iii in range(len(curr_layer)) if curr_layer[iii] != "" and self.__primitive_set.is_primitive(curr_layer[iii])]
            for i in range(len(elem_ind)):
                curr_ind = elem_ind[i]
                children = []
                start_ind = self.max_degree() * curr_ind
                end_ind = start_ind + self.max_degree()
                for j in range(start_ind, end_ind):
                    child = next_layer[j]
                    if next_layer[j] != "":
                        if isinstance(child, str) and not(Primitive.check_valid_primitive_name(child)):
                            child = child.strip()
                            if not(re.search(r'^x\d+', child) is None):
                                children.append(x[int(child[1:])])
                            elif not(re.search(r'^[ce]\d+\s', child) is None):
                                children.append(self.__terminal_set.cast(child))
                            else:
                                children.append(child)
                        elif isinstance(child, str):
                            children.append(child.strip())
                        else:
                            children.append(child)
                tre[layer_ind][curr_ind] = self.__primitive_set.get_primitive(tre[layer_ind][curr_ind])(*children)
        return tre[0][0]
