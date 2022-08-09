import inspect
import math
import random

import pandas as pd

from genepro import node_impl
from genepro.node import Node
from genepro.node_impl import Constant, Feature, IfThenElse
from genepro.util import tree_from_prefix_repr
from genepro.variation import subtree_mutation


def complexify(tree: Node) -> Node:
    node_classes = [c[1] for c in inspect.getmembers(node_impl, inspect.isclass)]
    internal_nodes = list()
    for node_cls in node_classes:
        # handle Features and Constants separately (also, avoid base class Node and IfThenElse)
        if node_cls == Node or node_cls == Feature or node_cls == Constant or node_cls == IfThenElse:
            continue
        node_obj = node_cls()
        internal_nodes.append(node_obj)

    # 10 random constants and 10 features
    leaf_nodes = list()
    for i in range(10):
        leaf_nodes.append(Feature(i))
        leaf_nodes.append(Constant(random.random()))

    mutated_tree = tree
    while not mutated_tree.get_n_nodes() > tree.get_n_nodes() | mutated_tree.get_height() > tree.get_height():
        mutated_tree = subtree_mutation(tree_from_prefix_repr(str(tree.get_subtree())), internal_nodes, leaf_nodes)

    return mutated_tree


data_dir = "dataset\\"

df = pd.read_csv(data_dir + "FeynmanEquationsRegularized.csv")
ast_formulae = df['AST_formula'].tolist()
complexified_formulae = []
for formula in ast_formulae:
    parsed_tree = tree_from_prefix_repr(formula.replace("pi", str(math.pi)))
    complexified_tree = complexify(parsed_tree)
    complexified_formulae.append(str(complexified_tree.get_subtree()))

d = {'Formula': ast_formulae, 'Complexified_formula': complexified_formulae}
pd.DataFrame(data=d).to_csv(data_dir + "FeynmanEquationsWarmUp.csv")
