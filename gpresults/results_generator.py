import re

import pandas as pd
from sympy import latex, parse_expr

from threads.MlPieRun import MlPieRun


def latex_format(readable_repr: str) -> str:
    try:
        latex_repr = latex(parse_expr(readable_repr, evaluate=False))
    except TypeError:
        latex_repr = readable_repr
    return re.sub(r"(\.[0-9][0-9])(\d+)", r"\1", latex_repr)


folder = '../exps/test_results_gp_traditional_pop200'
dataset = 'heating'
model = 'elastic_model'
target_model = 'phi'

dataframes = []

for i in range(10):
    seed = 700 + i
    filename = folder + '/best-' + dataset + '-' + model + '-GPT_' + str(seed) + '.csv'
    dataframes.append(pd.read_csv(filename))

df = pd.concat(dataframes)
df.drop(columns=df.columns[0], axis=1, inplace=True)
df = df.rename(columns={"latex_tree": "readable_tree"})
df["latex_tree"] = df["readable_tree"].map(latex_format)
df["problem"] = dataset

target_file = 'C:/Users/giorg/PycharmProjects/ML-PIE/gpresults/' + dataset + '_' + target_model + '.csv'
df.to_csv(target_file, index=False)
