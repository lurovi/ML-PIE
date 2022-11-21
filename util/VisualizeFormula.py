import math

import numpy as np
import pandas as pd
import sympy
from typing import List, Dict

from sklearn.metrics import r2_score

from exps.SklearnDatasetPreProcesser import SklearnDatasetPreProcessor
from genepro.util import tree_from_prefix_repr, replace_specified_operators_with_mean_value_constants
from nsgp.evolution.ParetoFrontUtil import ParetoFrontUtil
from threads.MlPieRun import MlPieRun

from util.PicklePersist import PicklePersist
pd.options.display.max_columns = 999


class VisualizeFormula:

    @staticmethod
    def read_file(folder_results: str, folder_dataset: str,
                  file_type: str, dataset_name: str, groundtruth: str, warmup: str, starting_seed: int, split: int, num_repeats: int) -> List[pd.DataFrame]:
        base_path = folder_results+"/"+file_type+"-"+dataset_name+"-"+"counts"+"-"+groundtruth+"-"+"Random Sampler Online"+"-"+warmup+"-"+"GPSU"+"_"
        path_dict = {"heating": "../exps/benchmark/energyefficiency.xlsx", "cooling": "../exps/benchmark/energyefficiency.xlsx"}
        datasettt = SklearnDatasetPreProcessor.load_data(dataset_name, split, path_dict=path_dict)
        training, validation, test = datasettt["training"], datasettt["validation"], datasettt["test"]
        data = []
        for i in range(starting_seed, starting_seed + num_repeats):
            file = base_path+str(i)+"_"+str(split)
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
                df.at[j, "training_r2"] = r2_score(training[1], train_res)
                df.at[j, "validation_r2"] = r2_score(validation[1], val_res)
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
                                            starting_seed: int, num_splits: List[int]) -> str:
        s = ""
        num_rows_data = str(len(percentiles) * len(index_list))
        num_rows_perc = str(len(index_list))
        for dataset_name in dataset_names:
            data = []
            for n in num_splits:
                data.extend(VisualizeFormula.read_file(folder_results, folder_dataset, file_type,
                                          dataset_name, groundtruth,
                                          warmup, starting_seed=starting_seed,
                                              split=n,
                                          num_repeats=1))
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
                    tree = tree_from_prefix_repr(df.loc[ind]["parsable_tree"])
                    latex_formula = "$"+MlPieRun.safe_latex_format(tree)+"$"
                    s += " & \\num{" + str(round(df.loc[ind]["training_r2"], 2)) + "}" + " & " + "\\num{" + str(
                        round(df.loc[ind]["validation_r2"], 2)) + "}" + " & " + latex_formula + " \\\\"
                    s += "\n"
                    if j == len(index_list) - 1 and i != len(percentiles) - 1:
                        s += "\\cline{2-5}"
                        s += "\n"

        return s

    @staticmethod
    def create_symbol_function_dict(n_features: int = 20) -> Dict:
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
            #return "$"+sympy.latex(formula)+"$"
            return "$"+sympy.latex(eval(formula, d))+"$"


if __name__ == "__main__":
    starting_seed, num_repeats = 700, 10
    percentiles = [90, 50, 10]
    index_list = [0, 1, 2]

    print(VisualizeFormula.print_latex_table_with_tao_datasets("../exps/test_results_gp_simulated_user_dropout",
                                                               "../exps/benchmark", "best",
                                                               ["boston", "heating"],
                                                               "node_wise_weights_sum_1",
                                                               "Elastic model",
                                                               percentiles,
                                                               index_list,
                                                               starting_seed=starting_seed,
                                                               num_splits=[40, 41, 42]
                                                               ))
