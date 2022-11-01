from deeplearn.trainer.Trainer import Trainer
from genepro.node import Node
import torch
from exps.groundtruth.GroundTruthComputer import GroundTruthComputer
from nsgp.encoder.TreeEncoder import TreeEncoder
from nsgp.evaluation.TreeEvaluator import TreeEvaluator


class NeuralNetTreeEvaluator(TreeEvaluator):
    def __init__(self, encoder: TreeEncoder, trainer: Trainer, negate: bool = False):
        super().__init__()
        self.__encoder = encoder
        self.__trainer = trainer
        self.__sign = -1.0 if negate else 1.0

    def evaluate(self, tree: Node, **kwargs) -> float:
        encoded_tree = self.__encoder.encode(tree, True)
        tensor_encoded_tree = torch.from_numpy(encoded_tree).float().reshape(1, -1)
        return self.__sign * self.__trainer.predict(tensor_encoded_tree)[0][0][0].item()
