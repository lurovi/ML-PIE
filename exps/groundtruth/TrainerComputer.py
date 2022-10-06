from deeplearn.trainer.Trainer import Trainer
from genepro.node import Node

import torch
from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.encoder.TreeEncoder import TreeEncoder


class TrainerComputer(GroundTruthComputer):
    def __init__(self, encoder: TreeEncoder, trainer: Trainer):
        super().__init__()
        self.set_name("neural_net")
        self.__encoder = encoder
        self.__trainer = trainer

    def compute(self, tree: Node) -> float:
        encoding = torch.from_numpy(self.__encoder.encode(tree, True)).float().reshape(1, -1)
        return self.__trainer.predict(encoding)[0][0].item()
