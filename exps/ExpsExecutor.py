from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from deeplearn.neuralnet import MLPNet, TwoPointsCompareTrainer
from util.PicklePersist import PicklePersist
from util.TorchSeedWorker import TorchSeedWorker


class ExpsExecutor:

    @staticmethod
    def execute_experiment_nn_ranking(title, generator_data_loader, file_name_training, file_name_dataset, train_size, activation_func,
                                      final_activation_func, hidden_layer_sizes, device, max_epochs=20, batch_size=1000,
                                      optimizer_name="adam", momentum=0.9):
        accs, ftrs = [], []
        for _ in range(10):
            trees = PicklePersist.decompress_pickle(file_name_dataset)
            training = PicklePersist.decompress_pickle(file_name_training)["training"]
            validation, test = trees["validation"], trees["test"]
            input_layer_size = len(validation[0][0])
            output_layer_size = 1
            trainloader = DataLoader(Subset(training, list(range(train_size))), batch_size=batch_size, shuffle=True,
                                     worker_init_fn=TorchSeedWorker.seed_worker,
                                     generator=generator_data_loader)
            valloader = DataLoader(Subset(validation, list(range(500))), batch_size=batch_size, shuffle=True,
                                   worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            # valloader = DataLoader(validation.remove_ground_truth_duplicates(), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            # trainloader_original = DataLoader(Subset(training.to_simple_torch_dataset(), list(range(train_size * 2))), batch_size=batch_size, shuffle=True, worker_init_fn=TorchSeedWorker.seed_worker, generator=generator_data_loader)
            net = MLPNet(activation_func, final_activation_func, input_layer_size, output_layer_size,
                         hidden_layer_sizes, dropout_prob=0.25)
            trainer = TwoPointsCompareTrainer(net, device, trainloader, verbose=False, max_epochs=max_epochs)
            trainer.train()
            # eval_val = trainer.evaluate_ranking(valloader)
            # eval_train = trainer.evaluate_ranking(trainloader_original)
            # print(title, " - Spearman Footrule on Training Set - ", eval_train)
            # print(title, " - Spearman Footrule on Validation Set - ", eval_val)
            # print("Accuracy: ", trainer.evaluate_classifier(valloader))
            accs.append(trainer.evaluate_classifier(valloader)[0])
            ftrs.append(trainer.evaluate_ranking(valloader))
        print(title)
        return sum(accs) / float(len(accs)), sum(ftrs) / float(len(ftrs))

    @staticmethod
    def example_execution_1(generator_data_loader, device):
        for target in ["weights_average", "weights_sum"]:
            for i in range(1, 5):
                i_str = str(i)
                for representation in ["counts", "onehot"]:
                    for final_activation in [nn.Identity(), nn.Sigmoid(), nn.Tanh()]:
                        print(ExpsExecutor.execute_experiment_nn_ranking(
                            target + " " + i_str + " " + representation,
                            generator_data_loader,
                            "data/" + representation + "_" + target + "_" + "trees_twopointscompare" + "_" + i_str + ".pbz2",
                            "data/" + representation + "_" + target + "_" + "trees" + "_" + i_str + ".pbz2", 200,
                            nn.ReLU(), final_activation,
                            [220, 140, 80, 26], device, max_epochs=1, batch_size=1
                        ))
