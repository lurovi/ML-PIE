from pymoo.core.evaluator import Evaluator


class CachingEvaluator(Evaluator):
    def __init__(self,
                 skip_already_evaluated: bool = True,
                 evaluate_values_of: list = ["F", "G", "H"],
                 callback=None):
        super(CachingEvaluator, self).__init__(skip_already_evaluated=skip_already_evaluated,
                                               evaluate_values_of=evaluate_values_of,
                                               callback=callback)

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):

        # get the design space value from the individuals
        X = pop.get("X")

        # call the problem to evaluate the solutions passing the old fitness values (if available)
        F = pop.get("F")
        out = problem.evaluate(X, return_values_of=evaluate_values_of, return_as_dictionary=True, fitness=F)

        # for each of the attributes set it to the problem
        for key, val in out.items():
            if val is not None:
                pop.set(key, val)

        # finally set all the attributes to be evaluated for all individuals
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))
