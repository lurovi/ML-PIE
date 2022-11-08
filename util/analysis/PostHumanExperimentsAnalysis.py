import os
import pandas as pd

from genepro.util import tree_from_prefix_repr
from util.PicklePersist import PicklePersist

folder = '../humanresults'

best_dataframes = []
trainers = []

for file in os.listdir(folder):
    if file.startswith("best"):
        best_dataframes.append(pd.read_csv(file))
    if file.startswith("trainer"):
        trainers.append(PicklePersist.decompress_pickle(file))

best_dataframe = pd.concat(best_dataframes)
parsable_trees = best_dataframe['parsable_tree'].tolist()
trees = []
for parsable_tree in parsable_trees:
    trees.append(tree_from_prefix_repr(parsable_tree))
