import pandas as pd

from util.python_math_parser.parser import Parser
from util.python_math_parser.tokenizer import Tokenizer


def protected_replace(string: str, values: dict, protected_subs: list):
    escape_char = "\\"
    split_char = '~'
    escaped_string = string
    for protected_sub in protected_subs:
        escaped_string = escaped_string.replace(protected_sub, split_char + escape_char + protected_sub + split_char)
    tokens = escaped_string.split(split_char)
    substituted_tokens = []
    for token in tokens:
        if token.startswith(escape_char):
            substituted_tokens.append(token[1:])
        else:
            substituted_token = token

            for key in sorted(values, key=lambda k: len(k), reverse=True):
                substituted_token = substituted_token.replace(key, values[key])
            substituted_tokens.append(substituted_token)
    return "".join(substituted_tokens)


data_dir = "D:\\Research\\ML-PIE\\"

df = pd.read_csv(data_dir + "FeynmanEquations.csv")
df = df.iloc[:100]

functions = ["+", "-", "*", "/", "sqrt", "exp", "log", "sin", "cos", "arcsin", "tanh"]

formula_column = "Formula"
variables_columns = []

for i in range(1, 10):
    variables_columns.append("v" + str(i) + "_name")

for col in variables_columns:
    df[col] = df[col].astype(str)

df[formula_column] = df[formula_column].astype(str)

renamed_formulae = []
ast_formulae = []

for index, row in df.iterrows():
    formula = row[formula_column]
    substitutions = {}
    for i in range(len(variables_columns)):
        variable_column = variables_columns[i]
        substitutions[row[variable_column]] = "x_" + str(i)

    formula = formula.replace("ln", "log")
    formula = protected_replace(formula, substitutions, functions)

    print(formula)

    token_iter = Tokenizer.token_iter("y=" + formula)
    PSR = Parser(token_iter)
    ast_formula = PSR.parse()

    renamed_formulae.append(formula)
    ast_formulae.append(ast_formula)

df["Standardized_formula"] = renamed_formulae
df["AST_formula"] = ast_formulae

df.to_csv(data_dir + "FeynmanEquationsRegularized.csv")
