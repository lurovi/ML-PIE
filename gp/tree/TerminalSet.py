from typing import List, Any
from gp.tree.Constant import Constant
from gp.tree.Ephemeral import Ephemeral
import random


class TerminalSet:
    def __init__(self, feature_types: List[Any], constants: List[Constant], ephemeral: List[Ephemeral]):
        self.__num_features = len(feature_types)
        self.__num_constants = len(constants)
        self.__num_ephemeral = len(ephemeral)
        self.__feature_types = feature_types
        self.__constants = constants
        self.__ephemeral = ephemeral

        self.__all_obj = feature_types + self.__constants + self.__ephemeral
        self.__all_types = feature_types + [c.type() for c in self.__constants] + [e.type() for e in self.__ephemeral]
        self.__all_idx = ["x" + str(i) for i in range(self.__num_features)] + \
                         ["c" + str(i) + " " + str(self.__constants[i]()) for i in range(self.__num_constants)] + \
                         ["e" + str(i) + " " for i in range(self.__num_ephemeral)]

    def is_there_type(self, provided_type: Any):
        candidates = [i for i in range(len(self.__all_types)) if self.__all_types[i] == provided_type]
        return bool(candidates)

    def all_idx(self):
        return ["x" + str(i) for i in range(self.__num_features)] + \
               ["c" + str(i) for i in range(self.__num_constants)] + \
               ["e" + str(i) for i in range(self.__num_ephemeral)]

    def get_type(self, idx):
        return self.__all_types[idx]

    def __str__(self):
        return f"N. Features: {self.__num_features} - N. Constants: {self.__num_constants} - N. Ephemeral: {self.__num_ephemeral}."

    def num_features(self):
        return self.__num_features

    def num_constants(self):
        return self.__num_constants

    def num_ephemeral(self):
        return self.__num_ephemeral

    def get_constant_ephemeral(self, s_idx: str) -> Any:
        end_ind = s_idx.find(" ")
        ind = int(s_idx[1:end_ind])
        if s_idx[0] == "c":
            return self.__all_obj[self.__num_features + ind]
        elif s_idx[0] == "e":
            return self.__all_obj[self.__num_features + self.__num_constants + ind]
        elif s_idx[0] == "x":
            return self.__all_obj[ind]

    def cast(self, s_idx: str) -> Any:
        objc = self.get_constant_ephemeral(s_idx)
        start_ind = s_idx.find(" ")
        return objc.cast(s_idx[(start_ind + 1):])

    @staticmethod
    def feature_id(s_idx: str) -> int:
        return int(s_idx[1:])

    def sample(self) -> str:
        ind = random.randint(0, len(self.__all_obj) - 1)
        return self.__extract(ind)

    def sample_typed(self, provided_type: Any) -> str:
        candidates = [i for i in range(len(self.__all_types)) if self.__all_types[i] == provided_type]
        if not candidates:
            raise LookupError(
                f"In the terminal set there is no type {str(provided_type)} available as type of one of the terminals in the set.")
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__extract(ind)

    def __extract(self, ind: int) -> str:
        if self.__all_idx[ind][0] == "x" or self.__all_idx[ind][0] == "c":
            return self.__all_idx[ind]
        elif self.__all_idx[ind][0] == "e":
            return self.__all_idx[ind] + str(self.__all_obj[ind]())
