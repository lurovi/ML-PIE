import pandas as pd
import numpy as np
from pymoo.indicators.hv import HV

use_ground_truth = True

folder = 'D://Research//ML-PIE//gp_simulated//'

encodings = ['counts']
datasets = ['boston', 'heating']
ground_truths = ['elastic_model', 'n_nodes', 'node_wise_weights_sum_1']
active_learnings = ['Random Sampler Online', 'Uncertainty Sampler Online']
warm_ups = ['Elastic model']
optimization_seeds = range(700, 710)
split_seeds = [40, 41, 42]

columns_to_keep = ['accuracy', 'interpretability', 'ground_truth_value']

hypervolume_dataframes = []

ref_point = np.array([1.1, 1.1])

for encoding in encodings:
    for dataset in datasets:
        for ground_truth in ground_truths:
            for active_learning in active_learnings:
                for warm_up in warm_ups:
                    for optimization_seed in optimization_seeds:
                        for split_seed in split_seeds:
                            name = 'best-' + dataset + '-' + encoding + '-' + ground_truth + '-' \
                                   + active_learning + '-' + warm_up + '-GPSU_' + str(optimization_seed) + '_' \
                                   + str(split_seed)
                            df_stochastic = \
                                pd.read_csv(folder + 'test_results_gp_simulated_user_dropout//' + name + '.csv')[
                                    columns_to_keep]
                            df_stochastic['profile'] = 'stochastic'
                            df_oracle = pd.read_csv(
                                folder + 'test_results_gp_simulated_user_dropout//'
                                + name.replace('best', 'bestrerun') + '_rerun.csv')[columns_to_keep]
                            df_oracle['profile'] = 'oracle'
                            df_constant = \
                                pd.read_csv(folder + 'test_results_gp_simulated_user_constant_rate//' + name + '.csv')[
                                    columns_to_keep]
                            df_constant['profile'] = 'constant'

                            # add here the lazy start and lazy end

                            df = pd.concat([df_stochastic, df_oracle, df_constant])
                            df['normalized_interpretability'] = (df.interpretability - df.interpretability.min()) / (
                                    df.interpretability.max() - df.interpretability.min())
                            df['normalized_accuracy'] = (df.accuracy - df.accuracy.min()) / (
                                    df.accuracy.max() - df.accuracy.min())
                            df['normalized_ground_truth_value'] = (
                                                                          df.ground_truth_value - df.ground_truth_value.min()) / (
                                                                          df.ground_truth_value.max() - df.ground_truth_value.min())

                            hypervolumes = dict()

                            for run in df['profile'].unique():
                                df_sub = df[df['profile'] == run]
                                normalized_accuracy = df_sub['normalized_accuracy'].tolist()
                                normalized_interpretability = df_sub['normalized_interpretability'].tolist()
                                normalized_ground_truth_value = df_sub['normalized_ground_truth_value'].tolist()
                                points_i = np.array([np.array(list(a)) for a in
                                                     zip(normalized_accuracy, normalized_interpretability)])
                                points_gt = np.array([np.array(list(a)) for a in
                                                      zip(normalized_accuracy, normalized_ground_truth_value)])
                                ind = HV(ref_point=ref_point)
                                hv_i = ind(points_i)
                                hypervolumes['i_' + run] = hv_i
                                hv_gt = ind(points_gt)
                                hypervolumes['gt_' + run] = hv_gt

                            hypervolume_df = pd.DataFrame(data=[hypervolumes])
                            hypervolume_df['encoding'] = encoding
                            hypervolume_df['dataset'] = dataset
                            hypervolume_df['ground_truth'] = ground_truth
                            hypervolume_df['active_learning'] = active_learning
                            hypervolume_df['warm_up'] = warm_up
                            hypervolume_df['optimization_seed'] = optimization_seed
                            hypervolume_df['split_seed'] = split_seed

                            hypervolume_dataframes.append(hypervolume_df)

hypervolume_dataframe = pd.concat(hypervolume_dataframes).reset_index(inplace=False, drop=True)
hypervolume_dataframe.to_csv('D://Research//ML-PIE//engagement_hypervolumes.csv', index=False)
