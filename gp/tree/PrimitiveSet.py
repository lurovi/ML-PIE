from typing import List, Any
import random
from gp.tree.Primitive import Primitive


class PrimitiveSet:
    def __init__(self, primitives: List[Primitive], return_type: Any):
        self.__primitive_dict = {p.name(): p for p in primitives}
        self.__primitive_idx = {primitives[i].name(): i for i in range(len(primitives))}
        self.__primitives = primitives
        self.__primitive_names = [p.name() for p in primitives]
        self.__num_primitives = len(primitives)
        self.__return_type = return_type
        self.__max_arity = max([p.arity() for p in primitives])

    def is_there_type(self, provided_type: Any):
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == provided_type]
        return bool(candidates)

    def change_return_type(self, new_return_type: Any):
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == new_return_type]
        if not candidates:
            raise AttributeError("You can't create a new primitive set with the provided new return type since there is no primitive in the current set that has the new return type as return type.")
        return PrimitiveSet(self.__primitives, new_return_type)

    def __str__(self):
        return f"N. Primitives: {self.__num_primitives} - Return type: {self.__return_type}."

    def __len__(self):
        return self.__num_primitives

    def max_arity(self):
        return self.__max_arity

    def primitive_names(self):
        return self.__primitive_names

    def num_primitives(self):
        return self.__num_primitives

    def return_type(self):
        return self.__return_type

    def get_primitive(self, name: str) -> Primitive:
        return self.__primitive_dict[name]

    def get_primitive_idx(self, name: str) -> Primitive:
        return self.__primitive_idx[name]

    def sample(self) -> Primitive:
        ind = random.randint(0, len(self.__primitives) - 1)
        return self.__primitives[ind]

    def sample_root(self) -> Primitive:
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == self.__return_type]
        if not(candidates):
            raise LookupError(f"In the primitive set there is no type {str(self.__return_type)} available as return type of one of the primitives in the set that can be used to sample a root primitive.")
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def sample_typed(self, provided_type: Any) -> Primitive:
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].return_type() == provided_type]
        if not(candidates):
            raise LookupError(f"In the primitive set there is no type {str(provided_type)} available as return type of one of the primitives in the set.")
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def sample_parameter_typed(self, provided_types: Any) -> Primitive:
        candidates = [i for i in range(len(self.__primitives)) if self.__primitives[i].parameter_types() == provided_types]
        if not(candidates):
            raise LookupError(f"In the primitive set there is no type {str(provided_types)} available.")
        ind = random.randint(0, len(candidates) - 1)
        ind = candidates[ind]
        return self.__primitives[ind]

    def is_primitive(self, s_idx: str) -> bool:
        names = [p.name() for p in self.__primitives]
        return s_idx in names
