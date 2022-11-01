from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival, binary_tournament
from pymoo.core.population import Population
from pymoo.docs import parse_doc_string
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.display.multi import MultiObjectiveOutput


# =========================================================================================================
# Implementation
# =========================================================================================================
from nsgp.evaluation.CachingEvaluator import CachingEvaluator


class NSGP2(NSGA2):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=binary_tournament),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowdingSurvival(),
                 output=MultiObjectiveOutput(),
                 re_evaluate=True,
                 **kwargs):
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            **kwargs)

        self.re_evaluate = re_evaluate
        self.evaluator = CachingEvaluator()

    def next(self):

        # get the infill solutions
        infills = self.infill()

        # call the advance with them after evaluation
        if infills is not None:
            if not self.re_evaluate or not self.is_initialized:
                self.evaluator.eval(self.problem, infills, algorithm=self)
            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()

    def _advance(self, infills=None, **kwargs):

        # the current population
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)
            if self.re_evaluate:
                self.evaluator.eval(self.problem, pop, algorithm=self, skip_already_evaluated=False)

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs)


parse_doc_string(NSGP2.__init__)
