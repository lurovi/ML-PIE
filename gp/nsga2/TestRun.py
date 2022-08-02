import math
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

from gp.nsga2.DuplicateTreeElimination import DuplicateTreeElimination
from gp.nsga2.SampleProblem import SampleProblem
from gp.nsga2.TreeCrossover import TreeCrossover
from gp.nsga2.TreeMutation import TreeMutation
from gp.nsga2.TreeSampling import TreeSampling
from gp.operator.OnePointCrossover import OnePointCrossover
from gp.operator.UniformMutation import UniformMutation
from gp.tree.Constant import Constant
from gp.tree.HalfHalfGenerator import HalfHalfGenerator
from gp.tree.Primitive import Primitive
from gp.tree.PrimitiveSet import PrimitiveSet
from gp.tree.TerminalSet import TerminalSet

# define tree structure and create tree generator, tree crossover and tree mutation
primitives = [Primitive("+", float, [float, float], lambda a, b: a + b),
              Primitive("-", float, [float, float], lambda a, b: a - b),
              Primitive("*", float, [float, float], lambda a, b: a * b),
              Primitive("max", float, [float, float], lambda a, b: max(a, b)),
              Primitive("min", float, [float, float], lambda a, b: min(a, b)),
              Primitive("^2", float, [float], lambda a: a ** 2),
              Primitive("/2", float, [float], lambda a: a / 2.0),
              Primitive("sin", float, [float], lambda a: math.sin(a)),
              Primitive("cos", float, [float], lambda a: math.cos(a))
              ]
constants = [Constant("2", 2.0), Constant("1", 1.0)]
ephemeral = []
primitive_set = PrimitiveSet(primitives, float)
terminal_set = TerminalSet([float], constants, ephemeral)
tree_sampling = TreeSampling(HalfHalfGenerator(primitive_set, terminal_set, 2, 6))

tree_crossover = TreeCrossover(OnePointCrossover())

tree_mutation = TreeMutation(UniformMutation())

algorithm = NSGA2(pop_size=10,
                  sampling=tree_sampling,
                  crossover=tree_crossover,
                  mutation=tree_mutation,
                  eliminate_duplicates=DuplicateTreeElimination())

res = minimize(SampleProblem(),
               algorithm,
               ('n_gen', 5),
               seed=1,
               verbose=False)

Scatter().add(res.F).show()
