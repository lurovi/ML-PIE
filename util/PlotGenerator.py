from functools import partial
from typing import Dict, List, Any

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import partial
import random

from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.structure.TreeStructure import TreeStructure
from util.EvaluationMetrics import EvaluationMetrics

from util.Sort import Sort


class PlotGenerator:

    @staticmethod
    def merge_dictionaries_of_list(dicts: List[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
        df = dicts[0]
        for i in range(1, len(dicts)):
            df = {k: df[k] + dicts[i][k] for k in df.keys()}
        return df

    @staticmethod
    def concatenate_dataframe_rows(dataframes: List[pd.DataFrame]):
        return pd.concat(dataframes).reset_index(inplace=False, drop=True)

    @staticmethod
    def filter_dataframe_rows_by_column_values(df: pd.DataFrame, filters: Dict[str, List[str]]):
        condition: List[bool] = [True] * df.shape[0]
        for filter_k in filters.keys():
            condition = condition & (df[filter_k].isin(filters[filter_k]))
        return df[condition].reset_index(inplace=False, drop=True)

    @staticmethod
    def plot_line(df, x, y, hue, style, folder_path, img_name, pgfplot=False, figsize=(10, 6)):
        sns.set(rc={"figure.figsize": figsize})
        sns.set_style("white")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        ax = sns.lineplot(data=df, x=x, y=y, hue=hue, style=style, estimator=np.mean, ci=90)
        if pgfplot:
            matplotlib.use("pgf")
            matplotlib.rcParams.update(
                {"pgf.texsystem": "pdflatex", 'font.family': 'serif', 'font.size': 11, 'text.usetex': True,
                 'pgf.rcfonts': False, })
            plt.savefig(folder_path+"/"+img_name+".pgf")
        else:
            plt.savefig(folder_path+"/"+img_name+".png")
        return ax

    @staticmethod
    def plot_random_ranking(device, dataloader):
        def __random_comparator(point_1, point_2, p):  # here point is the ground truth label and p a probability
            if random.random() < p:
                return point_1 < point_2
            else:
                return not (point_1 < point_2)

        def __random_spearman(device, dataloader, p):
            y_true = []
            points = []
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(device).float(), labels.to(device).float().reshape((labels.shape[0], 1))
                for i in range(len(inputs)):
                    points.append(inputs[i])
                    y_true.append(labels[i][0].item())
            y_true_2 = [x for x in y_true]
            y_true, _ = Sort.heapsort(y_true, lambda x, y: x < y, inplace=False, reverse=False)
            comparator = partial(__random_comparator, p=p)
            y_pred, _ = Sort.heapsort(y_true_2, comparator, inplace=False, reverse=False)
            return EvaluationMetrics.spearman_footrule(y_true, y_pred, lambda x, y: x == y)

        df = {"Probability": [], "Footrule": []}
        for p in np.arange(0, 1.1, 0.1):
            df["Probability"].append(p)
            ll = sum([__random_spearman(device, dataloader, p) for _ in range(20)]) / 20.0
            df["Footrule"].append(ll)
        plot = sns.lineplot(data=df, x="Probability", y="Footrule")
        plt.show()
        return plot

    @staticmethod
    def plot_encoding_size():
        P = [12, 16]
        V = [7, 9]
        B = [2, 3]
        L = [6, 7]
        E = ["Counts", "Level-wise Counts", "One-hot"]

        df = {"Encoding": [], "P-V": [], "B-L": [], "Size": []}
        for p in P:
            for v in V:
                for b in B:
                    for l in L:
                        cou, levcou, onh = TreeEncoder.encoding_size(p, v, b, l)
                        df["Encoding"].extend(E)
                        df["P-V"].extend([str(p) + "-" + str(v)] * 3)
                        df["B-L"].extend([str(b) + "-" + str(l)] * 3)
                        df["Size"].extend([cou, levcou, onh])
        sns.set(rc={"figure.figsize": (20, 20)})
        sns.set_style("white")
        df = pd.DataFrame(df)
        g = sns.catplot(data=df, x="Encoding", y="Size", col="P-V", hue="B-L", kind="bar", legend=True,
                        palette="colorblind")
        g.set(yscale='log')
        plt.show()
        return g


if __name__ == "__main__":
    df = {"a": ["AS", "AS", "DEF", "DER", "AS"], "b": ["GT", "ER", "ER", "GT", "OL"], "c": ["WE", "WE", "QQ", "WW", "QQ"]}
    df = pd.DataFrame(df)
    print(df.head())

    df_1 = PlotGenerator.filter_dataframe_rows_by_column_values(df, {"a": ["AS"], "c": ["WE"]})
    print(df_1.head())

    print(PlotGenerator.concatenate_dataframe_rows([df, df_1]).head(9))
