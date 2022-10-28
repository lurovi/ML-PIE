from functools import partial

import torch

import os

os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import torch.multiprocessing as mp

from exps.DatasetGenerator import DatasetGenerator
from exps.ExpsUtil import ExpsUtil

from nsgp.sampling.UncertaintyChooserOnlineFactory import UncertaintyChooserOnlineFactory
from nsgp.sampling.RandomChooserOnlineFactory import RandomChooserOnlineFactory

from threads.GPSimulatedUserExpsExecutor import GPSimulatedUserExpsExecutor

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

mp.set_sharing_strategy('file_system')


if __name__ == "__main__":
    #exit(1)
    # Setting torch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_dict = {"heating": "benchmark/energyefficiency.xlsx", "cooling": "benchmark/energyefficiency.xlsx"}
    pop_size = 200
    num_gen = 50
    starting_seed = 700
    num_repeats = 10
    idx = 1
    folder_name = "test_results_gp_simulated_user"
    for split_seed in [40, 41, 42]:
        for data_path_file in ["heating", "boston"]:
            structure, ground_truths, dataset, duplicates_elimination_little_data = ExpsUtil.create_structure(data_path_file, split_seed=split_seed, path_dict=path_dict)
            data_generator: DatasetGenerator = ExpsUtil.create_dataset_generator_with_warmup(folder_name, data_path_file,
                                                                                    structure, ground_truths)
            runner: GPSimulatedUserExpsExecutor = GPSimulatedUserExpsExecutor(folder_name,
                                                                              data_path_file,
                                                                              split_seed=split_seed,
                                                                              structure=structure,
                                                                              ground_truths=ground_truths,
                                                                              dataset=dataset,
                                                                              duplicates_elimination_little_data=duplicates_elimination_little_data,
                                                                              device=device,
                                                                              data_generator=data_generator,
                                                                              verbose=False)
            for encoding_type_str in ["counts"]:
                for ground_truth_str in ["node_wise_weights_sum"+"_"+str(idx), "elastic_model", "n_nodes"]:
                    for sampler_factory in [RandomChooserOnlineFactory(), UncertaintyChooserOnlineFactory()]:
                        for warmup in ["elastic_model"]:
                            pp = partial(runner.execute_gp_run, pop_size=pop_size, num_gen=num_gen,
                                                      encoding_type=encoding_type_str,
                                                      ground_truth_type=ground_truth_str,
                                                      sampler_factory=sampler_factory,
                                                      warmup=warmup)
                            pool = mp.Pool(num_repeats if mp.cpu_count() > num_repeats else (mp.cpu_count() - 1), maxtasksperchild=1)
                            _ = list(pool.map(pp, list(range(starting_seed, starting_seed + num_repeats))))
                            pool.close()
                            pool.join()
