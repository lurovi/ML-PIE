import pandas as pd
import numpy as np

tau = True

base_folder = 'D:/Research/ML-PIE/gp_simulated/test_results_gp_simulated_user_'
columns_to_keep = ['parsable_tree', 'latex_tree', 'accuracy', 'interpretability', 'ground_truth_value', 'seed',
                   'encoder_type', 'ground_truth_type', 'sampling', 'warm-up', 'data', 'split_seed', 'train_mse',
                   'test_mse', 'train_r2', 'test_r2']

target_folder = 'D:/Research/ML-PIE/'

percentiles = [0, 25, 50, 75, 100]

runs = ['original', 'rerun']
rates = ['constant_rate', 'lazy_end', 'lazy_start']
encodings = ['counts']
datasets = ['boston', 'heating']
ground_truths = ['elastic_model', 'n_nodes', 'node_wise_weights_sum_1']
active_learnings = ['Random Sampler Online']
warm_ups = ['Elastic model']
optimization_seeds = range(700, 710)
split_seeds = [40, 41, 42]

accuracy_dataframes = []
ground_truth_dataframes = []

for run in runs:
    for rate in rates:
        for encoding in encodings:
            for dataset in datasets:
                for ground_truth in ground_truths:
                    for active_learning in active_learnings:
                        for warm_up in warm_ups:
                            for optimization_seed in optimization_seeds:
                                for split_seed in split_seeds:
                                    filename = base_folder + rate + '/best-' + dataset + '-' + encoding + '-' + ground_truth + '-' + active_learning + '-' + warm_up + '-GPSU_' + str(
                                        optimization_seed) + '_' + str(split_seed) + '.csv'
                                    if run == 'rerun':
                                        filename = filename.replace('best', 'bestrerun')
                                        filename = filename.replace('.csv', '_rerun.csv')
                                    df = pd.read_csv(filename)
                                    df = df.iloc[:, 1:]
                                    df = df[columns_to_keep]
                                    df['front_size'] = len(df)
                                    df['rate'] = rate
                                    df['run'] = run
                                    accuracies = df['accuracy'].tolist()
                                    sorted_accuracies = accuracies.copy()
                                    sorted_accuracies.sort()
                                    ground_truth_values = df['ground_truth_value'].tolist()
                                    sorted_ground_truth_values = ground_truth_values.copy()
                                    sorted_ground_truth_values.sort()
                                    accuracy_indexes = []
                                    ground_truth_values_indexes = []
                                    items = list(range(0, len(df)))
                                    for p in percentiles:
                                        if tau:
                                            list_id = np.percentile(items, p, method='closest_observation')
                                            accuracy = sorted_accuracies[list_id]
                                            ground_truth_value = sorted_ground_truth_values[list_id]
                                        else:
                                            accuracy = np.percentile(accuracies, p, method='closest_observation')
                                            ground_truth_value = np.percentile(ground_truth_values, p,
                                                                               method='closest_observation')
                                        accuracy_indexes.append(accuracies.index(accuracy))
                                        ground_truth_values_indexes.append(
                                            ground_truth_values.index(ground_truth_value))
                                    df_accuracy = df.loc[accuracy_indexes, :]
                                    df_accuracy['accuracy_percentile'] = percentiles
                                    accuracy_dataframes.append(df_accuracy)
                                    df_ground_truths = df.loc[ground_truth_values_indexes, :]
                                    df_ground_truths['ground_truth_percentile'] = percentiles
                                    ground_truth_dataframes.append(df_ground_truths)

accuracy_dataframe = pd.concat(accuracy_dataframes).reset_index(inplace=False, drop=True)
accuracy_dataframe['tau'] = 'tau' if tau else 'percentile'
accuracy_dataframe.to_csv(target_folder + ('accuracy_tau.csv' if tau else 'accuracy.csv'), index=False)

ground_truth_dataframe = pd.concat(ground_truth_dataframes).reset_index(inplace=False, drop=True)
ground_truth_dataframe['tau'] = 'tau' if tau else 'percentile'
ground_truth_dataframe.to_csv(target_folder + ('ground_truth_tau.csv' if tau else 'ground_truth.csv'), index=False)
