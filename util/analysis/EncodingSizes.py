import pandas as pd

l_max = [4, 5, 6]
alpha = [2, 3]
M = 7
d = range(0, 50)

dataframes = []
for l in l_max:
    for a in alpha:
        for dd in d:
            n_counts = M + dd + 1 + 3
            n_lev_counts = l * (M + dd + 1) + 3
            n_one_hot = (a ** l - 1) * (M + dd + 1) / (a - 1)
            df = pd.DataFrame({
                'l_max': l,
                'alpha': a,
                'd': dd,
                'M': M,
                'n_counts': n_counts,
                'n_lev_counts': n_lev_counts,
                'n_one_hot': n_one_hot
            }, index=[0])
            dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True)
df.to_csv('D:/Research/ML-PIE/encoding_size.csv', index=False)
