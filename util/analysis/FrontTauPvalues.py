import pandas as pd
import scipy.stats as ss

filename = 'D:/Research/ML-PIE/ground_truth_tau.csv'

df = pd.read_csv(filename)
percentiles = range(0, 101, 25)

datasets = []
percs = []
trains = []
tests = []

for dataset in df['data'].unique():
    for percentile in percentiles:
        temp = df[(df.data == dataset) & (df.ground_truth_percentile == percentile)]
        dt_train = [temp.loc[ids, 'train_r2'].values for ids in temp.groupby('ground_truth_type').groups.values()]
        _, p_train = ss.kruskal(*dt_train)
        dt_test = [temp.loc[ids, 'test_r2'].values for ids in temp.groupby('ground_truth_type').groups.values()]
        _, p_test = ss.kruskal(*dt_test)
        datasets.append(dataset)
        percs.append(percentile)
        trains.append(p_train)
        tests.append(p_test)

p_df = pd.DataFrame(list(zip(datasets, percs, trains, tests)), columns=('data', 'percentile', 'train', 'test'))
p_df.to_csv('D:/Research/ML-PIE/percentiles_pvalues.csv', index=False)
