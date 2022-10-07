import random
from functools import partial
from typing import Dict

import numpy as np
import torch

import torch.multiprocessing as mp
from torch import nn

from deeplearn.model.MLPNet import MLPNet
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

from nsgp.sampling.UncertaintyChooserOnlineDistanceEmbeddingsFactory import \
    UncertaintyChooserOnlineDistanceEmbeddingsFactory
from nsgp.sampling.UncertaintyChooserOnlineFactory import UncertaintyChooserOnlineFactory
from nsgp.structure.TreeStructure import TreeStructure

from threads.GPSimulatedUserExpsExecutor import GPSimulatedUserExpsExecutor
from util.PicklePersist import PicklePersist


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
    mlp_net = MLPNet(nn.ReLU(), nn.Identity(), encoder.size(), 1, [220, 110, 25],
                     dropout_prob=0.25)
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
    return runner.run_minimization(seed)


if __name__ == "__main__":
    #exit(1)
    # Setting torch to use deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)
    # Setting the device in which data have to be loaded. It can be either CPU or GPU (cuda), if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pop_size = 210
    num_gen = 60
    starting_seed = 200
    num_repeats = 10
    idx = 1
    folder_name = "test_results_gp_traditional"
    #pool = mp.Pool(num_repeats if mp.cpu_count() > num_repeats else (mp.cpu_count() - 1))
    for data_path_file in ["california", "diabets", "windspeed", "friedman1", "vladislavleva", "boston"]:
        structure, ground_truths, dataset, duplicates_elimination_little_data = ExpsUtil.create_structure("benchmark/"+data_path_file+".pbz2")
        data_generator: DatasetGenerator = ExpsUtil.create_dataset_generator_with_warmup(folder_name, data_path_file,
                                                                                structure, ground_truths)
        second_fitness = {"elastic_model": MathElasticModelComputer(), "size": NumNodesNegComputer()}
        warmups = ["feynman", "elastic_model"]
        second_fitnesses = [None] + list(second_fitness.keys())
        encoders = {"counts": structure.get_encoder("counts")}
        for curr_second_fitness in second_fitnesses:
            if curr_second_fitness is not None:
                evaluators = [MSEEvaluator(dataset["training"][0], dataset["training"][1]),
                              GroundTruthEvaluator(second_fitness[curr_second_fitness], True)]
                runner: GPWithNSGA2 = GPWithNSGA2(structure, evaluators,
                                                  pop_size=pop_size, num_gen=num_gen,
                                                  duplicates_elimination_data=duplicates_elimination_little_data)
                pp = partial(runner.run_minimization, verbose=False, save_history=True, mutex=None)
                results = map(pp, list(range(starting_seed, starting_seed + num_repeats)))
                PicklePersist.compress_pickle(folder_name+"/"+data_path_file+"-"+curr_second_fitness, results)
                print("Executed "+data_path_file+" "+curr_second_fitness)
            else:
                for encoding_type in encoders.keys():
                    for warmup in warmups:
                        pp = partial(run_minimization_with_neural_net, pop_size=pop_size, num_gen=num_gen,
                                     duplicates_elimination_little_data=duplicates_elimination_little_data,
                                     dataset=dataset, structure=structure, encoder=encoders[encoding_type],
                                     encoding_type=encoding_type, warmup=warmup,
                                     data_generator=data_generator, device=device)
                        results = map(pp, list(range(starting_seed, starting_seed + num_repeats)))
                        PicklePersist.compress_pickle(folder_name + "/" + data_path_file + "-" + "neuralnet"+"-"+encoding_type+"-"+warmup, results)
                        print("Executed " + data_path_file + " " + "neuralnet"+" "+encoding_type+" "+warmup)
    #pool.close()
    #pool.join()
