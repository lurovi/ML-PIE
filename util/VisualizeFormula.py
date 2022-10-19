import math

import numpy as np
import pandas as pd
import sympy
from typing import List, Dict

from sklearn.metrics import r2_score

from genepro.util import tree_from_prefix_repr, replace_specified_operators_with_mean_value_constants
from nsgp.evolution.ParetoFrontUtil import ParetoFrontUtil

from util.PicklePersist import PicklePersist
pd.options.display.max_columns = 999


class VisualizeFormula:

    @staticmethod
    def read_file(folder_results: str, folder_dataset: str,
                  file_type: str, dataset_name: str, groundtruth: str, warmup: str, starting_seed: int, num_repeats: int) -> List[pd.DataFrame]:
        base_path = folder_results+"/"+file_type+"-"+dataset_name+"-"+"counts"+"-"+groundtruth+"-"+"Uncertainty Sampler Online"+"-"+warmup+"-"+"GPSU"+"_"
        datasettt = PicklePersist.decompress_pickle(folder_dataset + "/" + dataset_name + ".pbz2")
        training, validation, test = datasettt["training"], datasettt["validation"], datasettt["test"]
        data = []
        for i in range(starting_seed, starting_seed + num_repeats):
            file = base_path+str(i)
            if file_type == "nn":
                file += ".pth"
            else:
                file += ".csv"
            df = pd.read_csv(file)
            df.drop("Unnamed: 0", axis=1, inplace=True)
            df.rename(columns={"accuracy": "training_mse", "interpretability": "complexity"}, inplace=True)
            df = df.sort_values(by="training_mse", ascending=False)
            df.reset_index(inplace=True)
            df["tao"] = df.index + 1
            df["validation_mse"] = 1e+10
            df["training_r2"] = -1e+10
            df["validation_r2"] = -1e+10
            for j in range(df.shape[0]):
                tree = tree_from_prefix_repr(df.loc[j]["parsable_tree"])
                # tree = replace_specified_operators_with_mean_value_constants(tree, training[0], ["cos", "sin"])
                train_res, training_mse, slope, intercept = ParetoFrontUtil.find_slope_intercept_training(training[0], training[1], tree)
                val_res, validation_mse = ParetoFrontUtil.predict_validation_data(validation[0], validation[1], tree, slope, intercept)
                df.at[j, "training_r2"] = r2_score(train_res, training[1])
                df.at[j, "validation_r2"] = r2_score(val_res, validation[1])
                df.at[j, "validation_mse"] = validation_mse
                df.at[j, "training_mse"] = training_mse
                df.at[j, "parsable_tree"] = str(tree.get_subtree())
                df.at[j, "latex_tree"] = tree.get_readable_repr().replace("u-", "-")
            df = df[["tao", "parsable_tree", "latex_tree", "training_mse", "complexity", "validation_mse", "training_r2", "validation_r2"]]
            data.append(df)
        return data

    @staticmethod
    def print_latex_table_with_tao_datasets(folder_results: str, folder_dataset: str,
                                            file_type: str, dataset_names: List[str],
                                            groundtruth: str, warmup: str,
                                            percentiles: List[int], index_list: List[int],
                                            starting_seed: int, num_repeats: int) -> str:
        s = ""
        num_rows_data = str(len(percentiles) * len(index_list))
        num_rows_perc = str(len(index_list))
        for dataset_name in dataset_names:
            data = VisualizeFormula.read_file(folder_results, folder_dataset, file_type,
                                          dataset_name, groundtruth,
                                          warmup, starting_seed=starting_seed,
                                          num_repeats=num_repeats)
            s += "\n"
            s += "\\midrule"
            s += "\n"
            for i in range(len(percentiles)):
                perc = percentiles[i]
                for j in range(len(index_list)):
                    index = index_list[j]
                    if i == 0 and j == 0:
                        s += "\\multirow{"+num_rows_data+"}{*}{"+dataset_name[0].upper() + dataset_name[1:]+"}"
                    else:
                        s += " "
                    if j == 0:
                        s += " & \\multirow{"+num_rows_perc+"}{*}{"+str(perc)+"}"
                    else:
                        s += " & "
                    df = data[index]
                    ind = int(np.percentile(df["tao"], perc))
                    formula = df.loc[ind]["latex_tree"]
                    s += " & \\num{" + str(round(df.loc[ind]["training_r2"], 2)) + "}" + " & " + "\\num{" + str(
                        round(df.loc[ind]["validation_r2"], 2)) + "}" + " & " + VisualizeFormula.to_latex_eq(formula,
                                                                                                              simplify=False) + " \\\\"
                    s += "\n"
                    if j == len(index_list) - 1 and i != len(percentiles) - 1:
                        s += "\\cline{2-5}"
                        s += "\n"

        return s

    @staticmethod
    def create_symbol_function_dict(n_features: int = 50) -> Dict:
        d = {"x_" + str(i): sympy.Symbol("x_" + str(i)) for i in range(n_features)}
        #d["+"] = lambda x, y: x+y
        #d["-"] = lambda x, y: x-y
        #d["*"] = lambda x, y: x*y
        #d["/"] = lambda x, y: x/(abs(y) + 1e-9)
        #d["**"] = lambda x, y: (abs(x) + 1e-9) ** y
        #d["**2"] = lambda x: x ** 2
        #d["**3"] = lambda x: x ** 3
        d["log"] = lambda x: sympy.log(x)
        d["exp"] = lambda x: sympy.exp(x)
        d["sqrt"] = lambda x: sympy.sqrt(x)
        d["cos"] = lambda x: sympy.cos(x)
        d["sin"] = lambda x: sympy.sin(x)
        return d

    @staticmethod
    def to_latex_eq(formula: str, simplify: bool = False) -> str:
        d = VisualizeFormula.create_symbol_function_dict()
        if simplify:
            return "$"+sympy.latex(sympy.simplify(eval(formula, d)))+"$"
        else:
            return "$"+sympy.latex(eval(formula, d))+"$"


if __name__ == "__main__":
    starting_seed, num_repeats = 200, 10
    percentiles = [90, 60, 30]
    repeats_id = [4, 3, 6]

    print(VisualizeFormula.print_latex_table_with_tao_datasets("../exps/test_results_gp_simulated_user",
                                                               "../exps/benchmark", "best",
                                                               ["boston", "yachthydrodynamics", "friedman1"
                                                                ],
                                                               "node_wise_weights_sum_1",
                                                               "Elastic model",
                                                               percentiles,
                                                               repeats_id,
                                                               starting_seed=starting_seed,
                                                               num_repeats=num_repeats
                                                               ))
