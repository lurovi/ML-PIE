import random
from genepro.node import Node
import genepro
import numpy as np
from genepro.variation import generate_random_tree
from genepro.util import tree_from_prefix_repr, one_hot_encode_tree, counts_encode_tree
from nsgp.TreeGrammarStructure import TreeGrammarStructure

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    operators = [genepro.node_impl.Plus(), genepro.node_impl.Minus(), genepro.node_impl.Times(),
                 genepro.node_impl.Max(), genepro.node_impl.Min(),
                 genepro.node_impl.Square(), genepro.node_impl.Exp(),
                 genepro.node_impl.Cos(), genepro.node_impl.Sin(), genepro.node_impl.UnaryMinus(),
                 ]

    terminals = [genepro.node_impl.Feature(0), genepro.node_impl.Feature(1),
                 genepro.node_impl.Feature(2), genepro.node_impl.Feature(3),
                 genepro.node_impl.Constant()]

    print([str(op.symb) for op in operators])

    structure = TreeGrammarStructure(operators, 4, 5, 2)

    for _ in range(2):
        #a = generate_random_tree(operators, terminals, max_depth=5, curr_depth=0)
        a = structure.generate_tree()
    print(a)
    print(len(a))
    print(a.get_n_nodes())
    print(a.get_subtree())
    print(a.get_depth())
    print(a.get_height())
    a.get_readable_repr()
    print(a.get_dict_repr(max_arity=3, node_index=0))

    print(a(np.array([[2,3,1,4],[2,4,2,5],[7,5,8,3]])))

    '''
    oh = one_hot_encode_tree(a, [str(op.symb) for op in operators], n_features=4, max_depth=5,
                        max_arity=2)
    #print(tree_from_prefix_repr("+(x_0,x_1)").get_readable_repr())
    print(oh)
    print(sum(oh))
    co = counts_encode_tree(a, [str(op.symb) for op in operators], n_features=4, max_depth=5,
                        max_arity=2, additional_properties=True)
    print(co)
    print(sum(co[:15]))
    '''

    oh = structure.generate_one_hot_encoding(a)
    print(oh)
    print(sum(oh))
    co = structure.generate_counts_encoding(a, True)
    print(co)
    print(sum(co[:15]))
