from genepro.node import Node

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer


class NumNodesNegComputer(GroundTruthComputer):
    def __init__(self):
        super().__init__()
        self.set_name("n_nodes")

    def compute(self, tree: Node) -> float:
        return -1.0 * tree.get_n_nodes()
