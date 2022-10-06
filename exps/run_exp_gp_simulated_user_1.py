from functools import partial

import torch

import torch.multiprocessing as mp

from exps.DatasetGenerator import DatasetGenerator
from exps.ExpsUtil import ExpsUtil

from nsgp.sampling.UncertaintyChooserOnlineDistanceEmbeddingsFactory import \
    UncertaintyChooserOnlineDistanceEmbeddingsFactory
from nsgp.sampling.UncertaintyChooserOnlineFactory import UncertaintyChooserOnlineFactory

from threads.GPSimulatedUserExpsExecutor import GPSimulatedUserExpsExecutor


if __name__ == "__main__":
    exit(1)
    # Setting torch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pop_size = 210
    num_gen = 60
    starting_seed = 200
    num_repeats = 10
    idx = 1
    folder_name = "test_results_gp_simulated_user"
    for data_path_file in ["california", "diabets", "windspeed", "friedman1", "vladislavleva", "boston"]:
        structure, ground_truths, dataset, duplicates_elimination_little_data = ExpsUtil.create_structure("benchmark/"+data_path_file+".pbz2")
        data_generator: DatasetGenerator = ExpsUtil.create_dataset_generator_with_warmup(folder_name, data_path_file,
                                                                                structure, ground_truths)
        runner: GPSimulatedUserExpsExecutor = GPSimulatedUserExpsExecutor(folder_name,
                                                                          data_path_file,
                                                                          structure=structure,
                                                                          ground_truths=ground_truths,
                                                                          dataset=dataset,
                                                                          duplicates_elimination_little_data=duplicates_elimination_little_data,
                                                                          device=device,
                                                                          data_generator=data_generator)
        for encoding_type_str in ["counts", "level_wise_counts"]:
            for ground_truth_str in ["elastic_model", "node_wise_weights_sum"+"_"+str(idx)]:
                for sampler_factory in [UncertaintyChooserOnlineFactory(), UncertaintyChooserOnlineDistanceEmbeddingsFactory("max"), UncertaintyChooserOnlineDistanceEmbeddingsFactory("median")]:
                    for warmup in ["feynman", "elastic_model"]:
                        pp = partial(runner.execute_gp_run, pop_size=pop_size, num_gen=num_gen,
                                                  encoding_type=encoding_type_str,
                                                  ground_truth_type=ground_truth_str,
                                                  sampler_factory=sampler_factory,
                                                  warmup=warmup)
                        #for seed in range(starting_seed, starting_seed + num_repeats):
                        #    pp(seed)
                        pool = mp.Pool(num_repeats if mp.cpu_count() > num_repeats else (mp.cpu_count() - 1))
                        _ = pool.map(pp, list(range(starting_seed, starting_seed + num_repeats)))
                        pool.close()
                        pool.join()
