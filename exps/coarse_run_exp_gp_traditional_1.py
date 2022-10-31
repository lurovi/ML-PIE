import random
from functools import partial
from typing import Dict

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import torch

import torch.multiprocessing as mp
from torch import nn

from deeplearn.model.MLPNet import MLPNet
from deeplearn.model.DropOutMLPNet import DropOutMLPNet
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from deeplearn.trainer.TwoPointsCompareTrainerFactory import TwoPointsCompareTrainerFactory
from exps.DatasetGenerator import DatasetGenerator
from exps.ExpsUtil import ExpsUtil
from exps.groundtruth.MathElasticModelComputer import MathElasticModelComputer
from exps.groundtruth.NumNodesNegComputer import NumNodesNegComputer
from exps.groundtruth.TrainerComputer import TrainerComputer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.evaluation.GroundTruthEvaluator import GroundTruthEvaluator
from nsgp.evaluation.MSEEvaluator import MSEEvaluator
from nsgp.evolution.GPWithNSGA2 import GPWithNSGA2

from nsgp.structure.TreeStructure import TreeStructure


#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

mp.set_sharing_strategy('file_system')


def run_minimization_with_neural_net(seed: int, pop_size: int, num_gen: int,
                                     duplicates_elimination_little_data: np.ndarray,
                                     dataset: Dict,
                                     structure: TreeStructure,
                                     encoder: TreeEncoder,
                                     encoding_type: str, warmup: str,
                                     data_generator: DatasetGenerator, device: torch.device) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mlp_net = MLPNet(nn.ReLU(), nn.Tanh(), encoder.size(), 1, [150, 50])
    # mlp_net = DropOutMLPNet(nn.ReLU(), nn.Tanh(), encoder.size(), 1)
    interpretability_estimator = OnlineTwoPointsCompareTrainer(mlp_net, device,
                                                               warmup_trainer_factory=TwoPointsCompareTrainerFactory(
                                                                   False, 1),
                                                               warmup_dataset=data_generator.get_warm_up_data(
                                                                   encoding_type, warmup))
    evaluators = [MSEEvaluator(dataset["training"][0], dataset["training"][1]),
                  GroundTruthEvaluator(TrainerComputer(encoder, interpretability_estimator), True)]
    runner: GPWithNSGA2 = GPWithNSGA2(structure, evaluators,
                                      pop_size=pop_size, num_gen=num_gen,
                                      duplicates_elimination_data=duplicates_elimination_little_data)
    return runner.run_minimization(seed, verbose=True, save_history=False, mutex=None)


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
    num_repeats = 1
    idx = 1
    folder_name = "test_results_gp_traditional"
    #pool = mp.Pool(num_repeats if mp.cpu_count() > num_repeats else (mp.cpu_count() - 1), maxtasksperchild=1)
    for split_seed in [40, 41, 42]:
        for data_path_file in ["heating", "boston"]:
            structure, ground_truths, dataset, duplicates_elimination_little_data = ExpsUtil.create_structure(data_path_file, split_seed=split_seed, path_dict=path_dict)
            data_generator: DatasetGenerator = ExpsUtil.create_dataset_generator_with_warmup(folder_name, data_path_file,
                                                                                    structure, ground_truths)
            second_fitness = {"elastic_model": MathElasticModelComputer(), "n_nodes": NumNodesNegComputer()}
            warmups = ["elastic_model"]
            second_fitnesses = list(second_fitness.keys())
            encoders = {"counts": structure.get_encoder("counts")}
            for curr_second_fitness in second_fitnesses:
                if curr_second_fitness is not None:
                    evaluators = [MSEEvaluator(dataset["training"][0], dataset["training"][1]),
                                  GroundTruthEvaluator(second_fitness[curr_second_fitness], True)]
                    runner: GPWithNSGA2 = GPWithNSGA2(structure, evaluators,
                                                      pop_size=pop_size, num_gen=num_gen,
                                                      duplicates_elimination_data=duplicates_elimination_little_data)
                    pp = partial(runner.run_minimization, verbose=True, save_history=False, mutex=None)
                    results = list(map(pp, list(range(starting_seed, starting_seed + num_repeats))))

                    for res in results:
                        real_result, executionTimeInMinutes, curr_seed = res["result"], res["executionTimeInHours"]*60, res["seed"]
                        print(executionTimeInMinutes)
                        #ExpsUtil.save_pareto_fronts_from_result_to_csv(folder_name=folder_name,
                        #                                               result=real_result, seed=curr_seed, split_seed=split_seed,
                        #                                               pop_size=pop_size, num_gen=num_gen,
                        #                                               num_offsprings=pop_size,
                        #                                               dataset=data_path_file,
                        #                                               groundtruth=curr_second_fitness)

                    print("Executed "+data_path_file+" "+curr_second_fitness+" "+str(split_seed))
                    exit(1)
                else:
                    for encoding_type in encoders.keys():
                        for warmup in warmups:
                            pp = partial(run_minimization_with_neural_net, pop_size=pop_size, num_gen=num_gen,
                                         duplicates_elimination_little_data=duplicates_elimination_little_data,
                                         dataset=dataset, structure=structure, encoder=encoders[encoding_type],
                                         encoding_type=encoding_type, warmup=warmup,
                                         data_generator=data_generator, device=device)
                            results = list(map(pp, list(range(starting_seed, starting_seed + num_repeats))))

                            for res in results:
                                real_result, executionTimeInMinutes, curr_seed = res["result"], res["executionTimeInHours"]*60, res["seed"]
                                print(executionTimeInMinutes)
                                #ExpsUtil.save_pareto_fronts_from_result_to_csv(folder_name=folder_name,
                                #                                               result=real_result, seed=curr_seed, split_seed=split_seed,
                                #                                               pop_size=pop_size, num_gen=num_gen,
                                #                                               num_offsprings=pop_size,
                                #                                               dataset=data_path_file,
                                #                                               groundtruth="neuralnet"+"_"+encoding_type+"_"+warmup,
                                #                                               encoder_type=encoding_type,
                                #                                               warmup=warmup)

                            print("Executed " + data_path_file + " " + "neuralnet"+" "+encoding_type+" "+warmup+" "+str(split_seed))
                            exit(1)
    #pool.close()
    #pool.join()
