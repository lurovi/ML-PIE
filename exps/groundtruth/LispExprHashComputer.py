from genepro.node import Node

import zlib

from exps.groundtruth.GroundTruthComputer import GroundTruthComputer


class LispExprHashComputer(GroundTruthComputer):
    def __init__(self, char_encoding: str = "utf-8"):
        super().__init__()
        self.set_name("hash_value")
        self.__char_encoding = char_encoding

    def compute(self, tree: Node) -> float:
        lisp_expr = tree.get_string_as_lisp_expr()
        return zlib.adler32(bytes(lisp_expr, self.__char_encoding))
