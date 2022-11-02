import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

encodings = ['counts']
datasets = ['boston', 'heating']
ground_truths = ['elastic_model', 'n_nodes', 'node_wise_weights_sum_1']
active_learnings = ['Random Sampler Online', 'Uncertainty Sampler Online']
warm_ups = ['Elastic model']
optimization_seeds = range(700, 710)
split_seeds = [40, 41, 42]

columns_to_keep = ['accuracy', 'interpretability']

hypervolume_dataframes = []

ref_point = np.array([1.1, 1.1])

for encoding in encodings:
    for dataset in datasets:
        for ground_truth in ground_truths:
            for active_learning in active_learnings:
                for warm_up in warm_ups:
                    for optimization_seed in optimization_seeds:
                        for split_seed in split_seeds:
                            name = 'test_results_gp_simulated_user_dropout/best-' + dataset + '-' + encoding + '-' + ground_truth + '-' + active_learning + '-' + warm_up + '-GPSU_' + str(
                                optimization_seed) + '_' + str(split_seed)
                            original_filename = name + '.csv'
                            rerun_filename = name.replace('best', 'bestrerun') + '_rerun.csv'
                            df_original = pd.read_csv(original_filename)[columns_to_keep]
                            df_original['run'] = 'original'
                            df_rerun = pd.read_csv(rerun_filename)[columns_to_keep]
                            df_rerun['run'] = 'rerun'

                            df = pd.concat([df_original, df_rerun])
                            df['normalized_interpretability'] = (df.interpretability - df.interpretability.min()) / (
                                    df.interpretability.max() - df.interpretability.min())
                            df['normalized_accuracy'] = (df.accuracy - df.accuracy.min()) / (
                                    df.accuracy.max() - df.accuracy.min())

                            hypervolumes = dict()

                            for run in df['run'].unique():
                                df_sub = df[df['run'] == run]
                                normalized_accuracy = df_sub['normalized_accuracy'].tolist()
                                normalized_interpretability = df_sub['normalized_interpretability'].tolist()
                                points = np.array([np.array(list(a)) for a in
                                                   zip(normalized_accuracy, normalized_interpretability)])
                                ind = HV(ref_point=ref_point)
                                hv = ind(points)
                                hypervolumes[run] = [hv]

                            hypervolume_df = pd.DataFrame(data=hypervolumes)
                            hypervolume_df['encoding'] = encoding
                            hypervolume_df['dataset'] = dataset
                            hypervolume_df['ground_truth'] = ground_truth
                            hypervolume_df['active_learning'] = active_learning
                            hypervolume_df['warm_up'] = warm_up
                            hypervolume_df['optimization_seed'] = optimization_seed
                            hypervolume_df['split_seed'] = split_seed

                            hypervolume_dataframes.append(hypervolume_df)

hypervolume_dataframe = pd.concat(hypervolume_dataframes).reset_index(inplace=False, drop=True)
hypervolume_dataframe['delta'] = hypervolume_dataframe['rerun'] - hypervolume_dataframe['original']
hypervolume_dataframe.to_csv('hypervolumes.csv', index=False)
