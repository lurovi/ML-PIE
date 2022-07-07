from tree.Primitive import *
import numpy as np


def generate_full_tree(primitive_set, ret_type, num_features, max_depth, ephemeral_function = None):
    pass


if __name__ == "__main__":
    primitive_set = [Primitive("+", np.float32, [np.float32, np.float32], lambda x, y: x + y),
                     Primitive("-", np.float32, [np.float32, np.float32], lambda x, y: x - y),
                     Primitive("*", np.float32, [np.float32, np.float32], lambda x, y: x * y),
                     Primitive("^2", np.float32, [np.float32, np.float32], lambda x: x ** 2),
                     Primitive("*2", np.float32, [np.float32, np.float32], lambda x: x * 2.0),
                     Primitive("/2", np.float32, [np.float32, np.float32], lambda x: x / 2.0),
                     Primitive("*3", np.float32, [np.float32, np.float32], lambda x: x * 3.0),
                     Primitive("/3", np.float32, [np.float32, np.float32], lambda x: x / 3.0)
                     ]
    num_features = 10
    has_ephemeral = True
    max_depth = 5





