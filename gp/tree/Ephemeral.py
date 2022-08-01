from typing import Callable, Any


class Ephemeral:
    def __init__(self, name: str, func: Callable):
        self.__name = name
        self.__func = func
        self.__type = type(self.__func())

    def __str__(self):
        return f"{self.__name} : {self.__func} --> {self.__type}"

    def __call__(self):
        return self.__func()

    def name(self):
        return self.__name

    def type(self):
        return self.__type

    def cast(self, val: Any):
        return self.__type(val)
