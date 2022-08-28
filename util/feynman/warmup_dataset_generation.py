import inspect
import math
import random

import numpy as np
import pandas as pd

from genepro import node_impl
from genepro.node import Node
from genepro.node_impl import Constant, Feature, IfThenElse
from genepro.util import tree_from_prefix_repr
from genepro.variation import subtree_mutation
from sympy import parse_expr, latex

from util.TreeGrammarStructure import TreeGrammarStructure


def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


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
        leaf_nodes.append(Constant(truncate(random.random(), 2)))

    while True:
        mutated_tree = subtree_mutation(tree_from_prefix_repr(str(tree.get_subtree())), internal_nodes, leaf_nodes, max_depth=7)
        if mutated_tree.get_n_nodes() >= tree.get_n_nodes():
            break

    return mutated_tree


def formula_to_latex(math_formula):
    readable_formula = tree_from_prefix_repr(math_formula).get_readable_repr().replace("u-", "-").replace(
        "3.141592653589793", "pi")
    return latex(parse_expr(readable_formula, evaluate=False))


seed = 1
random.seed(seed)
np.random.seed(seed)

data_dir = "dataset\\"

df = pd.read_csv(data_dir + "FeynmanEquationsRegularized.csv")
ast_formulae = df['AST_formula'].tolist()

formulae = []
complexified_formulae = []

for formula in ast_formulae:
    formulae.append(formula.replace("pi", str(math.pi)))
    parsed_tree = tree_from_prefix_repr(formula.replace("pi", str(math.pi)))
    complexified_tree = complexify(parsed_tree)
    complexified_formulae.append(str(complexified_tree.get_subtree()))

latex_formulae = list(map(formula_to_latex, formulae))
latex_complexified_formulae = list(map(formula_to_latex, complexified_formulae))

d = {'Formula': formulae, 'Complexified_formula': complexified_formulae, 'Latex_formula': latex_formulae,
     'Latex_complexified_formula': latex_complexified_formulae}

pd.DataFrame(data=d).to_csv(data_dir + "FeynmanEquationsWarmUp.csv")
