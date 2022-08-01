from typing import Any


class Constant:
    def __init__(self, name: str, val: Any):
        self.__name = name
        self.__val = val
        self.__type = type(self.__val)

    def __str__(self):
        return f"{self.__name} : {self.__val} --> {self.__type}"

    def __call__(self):
        return self.__val

    def name(self):
        return self.__name

    def type(self):
        return self.__type

    def cast(self, val: Any):
        return self.__type(val)
