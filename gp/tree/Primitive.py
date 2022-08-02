from typing import Any, List, Callable
import re


class Primitive:
    def __init__(self, name: str, return_type: Any, parameter_types: List[Any], function: Callable):
        if not (Primitive.check_valid_primitive_name(name)):
            raise AttributeError(
                f"Invalid name. {name} is not a valid name for a primitive. Please avoid starting the name with either x or c or e followed by a number.")
        self.__name = name
        self.__arity = len(parameter_types)
        self.__return_type = return_type
        self.__parameter_types = parameter_types
        self.__function = function

    @staticmethod
    def check_valid_primitive_name(s: str) -> bool:
        return s.strip() != "" and re.search(r'^[xce]\d+', s) is None

    def __str__(self):
        return f"{self.__name} : {self.__parameter_types} --> {self.__return_type}"

    def __call__(self, *args):
        return self.__function(*args)

    def __len__(self):
        return self.__arity

    def arity(self):
        return self.__arity

    def name(self):
        return self.__name

    def return_type(self):
        return self.__return_type

    def parameter_types(self):
        return self.__parameter_types
