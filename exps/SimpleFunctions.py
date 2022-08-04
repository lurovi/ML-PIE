import math
import random


class SimpleFunctions:
    def __init__(self):
        pass

    @staticmethod
    def ephe_0():
        return random.random()

    @staticmethod
    def ephe_1():
        return float(random.randint(0, 4))

    @staticmethod
    def sum(a, b):
        return a + b

    @staticmethod
    def sub(a, b):
        return a - b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def max(a, b):
        return max(a, b)

    @staticmethod
    def min(a, b):
        return min(a, b)

    @staticmethod
    def abs(a):
        return abs(a)

    @staticmethod
    def neg(a):
        return -a

    @staticmethod
    def power2(a):
        return a ** 2

    @staticmethod
    def mulby2(a):
        return a * 2.0

    @staticmethod
    def divby2(a):
        return a / 2.0

    @staticmethod
    def cos(a):
        return math.cos(a)

    @staticmethod
    def sin(a):
        return math.sin(a)
