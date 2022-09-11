from genepro.node import Node

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer


class InterpretabilityShapeComputer(GroundTruthComputer):
    def __init__(self):
        super().__init__()
        self.set_name("int_shape")

    def compute(self, tree: Node) -> float:
        df = tree.tree_numerical_properties()
        return (df["height"]+1)/float(df["n_nodes"]) + df["max_arity"]/float(df["max_breadth"]) + df["n_leaf_nodes"]/float(df["n_nodes"])
