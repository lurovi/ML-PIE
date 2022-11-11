import os
import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV
import scipy.stats as ss

base_folder = 'D:/Research/ML-PIE/gp_simulated/test_results_gp_simulated_user_'
target_folder = 'D:/Research/ML-PIE/'

rates = ['constant_rate', 'lazy_end', 'lazy_start', 'oracle']
columns_to_keep = ['accuracy', 'interpretability', 'ground_truth_value', 'seed',
                   'encoder_type', 'ground_truth_type', 'sampling', 'warm-up', 'data', 'split_seed', 'train_mse',
                   'test_mse', 'train_r2', 'test_r2']

# merge together all fronts
dataframes = []
for rate in rates:
    filenames = os.listdir(base_folder + rate)
    for filename in filenames:
        if not filename.__contains__('rerun'):
            df = pd.read_csv(base_folder + rate + '/' + filename)
            df = df[columns_to_keep]
            df['rate'] = rate
            dataframes.append(df)
df = pd.concat(dataframes, ignore_index=True)

# reshape dataframe
df_train = df.copy()
df_train['phase'] = 'train'
df_train.drop(columns=['test_mse', 'test_r2'], inplace=True)
df_train.rename(columns={'train_mse': 'mse', 'train_r2': 'r2'}, inplace=True)
df_test = df.copy()
df_test['phase'] = 'test'
df_test.drop(columns=['train_mse', 'train_r2'], inplace=True)
df_test.rename(columns={'test_mse': 'mse', 'test_r2': 'r2'}, inplace=True)
df = pd.concat([df_train, df_test], ignore_index=True)

# normalize the values for each ground_truth_type and dataset
normalized_dataframes = []
columns_to_normalize = ['ground_truth_value']
for data in df['data'].unique():
    for ground_truth_type in df['ground_truth_type'].unique():
        p_df = df[(df.data == data) & (df.ground_truth_type == ground_truth_type)]
        for c in columns_to_normalize:
            p_df['normalized_' + c] = (p_df[c] - p_df[c].min()) / (p_df[c].max() - p_df[c].min())
        normalized_dataframes.append(p_df)
df = pd.concat(normalized_dataframes, ignore_index=True)
df.to_csv(target_folder + 'gp_simulated/fronts_merged.csv', index=False)

# for each front compute hypervolume
ind_r2 = HV(ref_point=np.array([1.1, 0.1]))
datas = []
ground_truth_types = []
split_seeds = []
seeds = []
rates = []
phases = []
hypervolumes = []
for data in df['data'].unique():
    for ground_truth_type in df['ground_truth_type'].unique():
        for split_seed in df['split_seed'].unique():
            for seed in df['seed'].unique():
                for rate in df['rate'].unique():
                    for phase in df['phase'].unique():
                        front = df[(df.data == data) & (df.ground_truth_type == ground_truth_type) & (
                                df.split_seed == split_seed) & (df.seed == seed) & (df.rate == rate) & (
                                           df.phase == phase)]
                        datas.append(data)
                        ground_truth_types.append(ground_truth_type)
                        split_seeds.append(split_seed)
                        seeds.append(seed)
                        rates.append(rate)
                        phases.append(phase)

                        gt = front['normalized_ground_truth_value'].tolist()
                        r2 = front['r2'].tolist()
                        negative_r2 = [-x for x in r2]
                        hypervolume = ind_r2(np.array([np.array(list(a)) for a in zip(gt, negative_r2)]))
                        hypervolumes.append(hypervolume)

df = pd.DataFrame(list(zip(datas, ground_truth_types, split_seeds, seeds, rates, phases, hypervolumes)),
                  columns=('data', 'ground_truth_type', 'split_seed', 'seed', 'rate', 'phase', 'hypervolume'))
df.to_csv(target_folder + 'hypervolumes.csv', index=False)

datas = []
ground_truth_types = []
phases = []
pvalues = []
for data in df['data'].unique():
    for ground_truth_type in df['ground_truth_type'].unique():
        for phase in df['phase'].unique():
            temp = df[(df.data == data) & (df.ground_truth_type == ground_truth_type) & (df.phase == phase)]
            dt = [temp.loc[ids, 'hypervolume'].values for ids in temp.groupby('rate').groups.values()]
            H, p = ss.kruskal(*dt)
            datas.append(data)
            ground_truth_types.append(ground_truth_type)
            phases.append(phase)
            pvalues.append(p)

p_df = pd.DataFrame(list(zip(datas, ground_truth_types, phases, pvalues)),
                    columns=('data', 'ground_truth_type', 'phase', 'pvalue'))
p_df.to_csv(target_folder + 'hypervolume_pvalues.csv', index=False)
