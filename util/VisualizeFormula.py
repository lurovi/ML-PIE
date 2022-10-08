import math

import pandas as pd
import sympy
from typing import List, Dict
from util.GenerateSymbols import GenerateSymbols
from util.PicklePersist import PicklePersist


class VisualizeFormula:

    @staticmethod
    def read_file(folder: str, file_type:str, dataset_name: str, groundtruth: str, warmup: str, starting_seed: int, num_repeats: int, num_gen: int) -> List[pd.DataFrame]:
        base_path = folder+"/"+file_type+"-"+dataset_name+"-"+"counts"+"-"+groundtruth+"-"+"Uncertainty Sampler Online"+"-"+warmup+"-"+"GPSU"+"_"
        data = []
        for i in range(starting_seed, starting_seed + num_repeats):
            file = base_path+str(i)
            if file_type == "nn":
                file += ".pth"
            else:
                file += ".csv"
            df = pd.read_csv(file)
            df.drop("Unnamed: 0", axis=1, inplace=True)
            df = df[df["generation"] == num_gen-1]
            df.reset_index(inplace=True)
            data.append(df)
        return data

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
    data = VisualizeFormula.read_file("../exps/test_results_gp_simulated_user", "best", "boston", "elastic_model",
                                      "Elastic model", 200, 10, 60)
    print(data[0].head())
    df = data[0]
    formula = df["latex_tree"][11]
    print(VisualizeFormula.to_latex_eq(formula, simplify=False))

