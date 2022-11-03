import re

import pandas as pd
from sympy import latex, parse_expr

from genepro.util import tree_from_prefix_repr
from threads.MlPieRun import MlPieRun

folder = '../exps/test_results_gp_traditional_dropout'
datasets = ['heating', 'boston']
models = ['elastic_model', 'n_nodes']
target_models = ['phi', 'size']

for dataset in datasets:
    for k in range(len(models)):
        model = models[k]
        target_model = target_models[k]
        for j in range(3):
            split_seed = 40 + j
            dataframes = []

            for i in range(10):
                seed = 700 + i
                filename = folder + '/best-' + dataset + '-' + model + '-GPT_' + str(seed) + '_' + str(
                    split_seed) + '.csv'
                dataframes.append(pd.read_csv(filename))

            df = pd.concat(dataframes)
            df.drop(columns=df.columns[0], axis=1, inplace=True)
            df = df.rename(columns={"latex_tree": "readable_tree"})
            df["latex_tree"] = df["parsable_tree"].map(tree_from_prefix_repr).map(MlPieRun.safe_latex_format)
            df["problem"] = dataset

            target_file = dataset + '_' + target_model + '_' + str(split_seed) + '.csv'
            df.to_csv(target_file, index=False)
