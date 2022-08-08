import pandas as pd

from genepro.util import tree_from_prefix_repr

data_dir = "D:\\Research\\ML-PIE\\"

df = pd.read_csv(data_dir + "FeynmanEquationsRegularized.csv")

original_formulae = df['Formula'].tolist()
ast_formulae = df['AST_formula'].tolist()

for i in range(100):
    pi_and_e_escaped = ast_formulae[i].replace("pi", "3.14")
    tree_string = pi_and_e_escaped.replace("y = ", "").replace("(", ",").replace(")", "").replace(" ", "")
    tree = tree_from_prefix_repr(tree_string)
