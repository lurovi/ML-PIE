import multiprocessing
import threading
from typing import Optional

import pandas as pd

from nsgp.callback.PopulationAccumulator import PopulationAccumulator
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.problem.RegressionProblemWithNeuralEstimate import RegressionProblemWithNeuralEstimate
from threads.MlPieRun import MlPieRun
from threads.OptimizationThread import OptimizationThread


class MlPieRunProcess(multiprocessing.Process):

    def __init__(self,
                 run_id,
                 seed,
                 problem,
                 dataset,
                 tree_encoder,
                 interpretability_estimator,
                 pair_chooser,
                 optimization_algorithm,
                 termination,
                 path,
                 timeout=20 * 60
                 ):
        super().__init__()
        self.run_id = run_id
        self.seed = seed
        self.dataset = dataset
        self.tree_encoder = tree_encoder
        self.interpretability_estimator = interpretability_estimator
        self.pair_chooser = pair_chooser
        self.optimization_algorithm = optimization_algorithm
        self.termination = termination
        self.path = path
        self.problem = problem
        self.timeout = timeout
        self.alert_pipe_parent, self.alert_pipe_child = multiprocessing.Pipe()
        self.request_pipe_parent, self.request_pipe_child = multiprocessing.Pipe()
        self.progress_pipe_parent, self.progress_pipe_child = multiprocessing.Pipe()
        self.feedback_pipe_parent, self.feedback_pipe_child = multiprocessing.Pipe()
        self.pareto_pipe_parent, self.pareto_pipe_child = multiprocessing.Pipe()

    def run(self) -> None:
        mutex = threading.Lock()
        regression_problem = RegressionProblemWithNeuralEstimate(self.dataset["training"][0],
                                                                 self.dataset["training"][1],
                                                                 mutex=mutex,
                                                                 tree_encoder=self.tree_encoder,
                                                                 interpretability_estimator=self.interpretability_estimator
                                                                 )
        population_storage = set()
        callback = PopulationAccumulator(population_storage=population_storage)
        optimization_thread = OptimizationThread(
            optimization_algorithm=self.optimization_algorithm,
            problem=regression_problem,
            termination=self.termination,
            seed=self.seed,
            callback=callback,
            verbose=False,
            save_history=False
        )
        interpretability_estimate_updater = InterpretabilityEstimateUpdater(individuals=population_storage, mutex=mutex,
                                                                            interpretability_estimator=self.interpretability_estimator,
                                                                            encoder=self.tree_encoder,
                                                                            pair_chooser=self.pair_chooser)
        ml_pie_run = MlPieRun(run_id=self.run_id,
                              optimization_thread=optimization_thread,
                              interpretability_estimate_updater=interpretability_estimate_updater,
                              path=self.path,
                              parameters={"problem": self.problem})
        ml_pie_run.start()
        while self.alert_pipe_child.poll(self.timeout):
            message = self.alert_pipe_child.recv()
            if message == 'join':
                ml_pie_run.join()
            elif message == 'request_models':
                models = ml_pie_run.request_models()
                self.request_pipe_child.send(models)
            elif message == 'request_progress':
                progress = ml_pie_run.request_progress()
                self.progress_pipe_child.send(progress)
            elif message == 'feedback':
                feedback = self.feedback_pipe_child.recv()
                feedback_response = ml_pie_run.provide_feedback(feedback)
                self.feedback_pipe_child.send(feedback_response)
            elif message == 'flush':
                ml_pie_run.flush()
            elif message == 'pareto':
                pareto_front = ml_pie_run.get_pareto_front()
                self.pareto_pipe_child.send(pareto_front)
                print(self.run_id + ' correctly ended')
                return
        print(self.run_id + ' timed out')

    def join(self, timeout: Optional[float] = ...) -> None:
        self.alert_pipe_parent.send("join")

    def request_models(self) -> dict:
        self.alert_pipe_parent.send("request_models")
        return self.request_pipe_parent.recv()

    def request_progress(self) -> float:
        self.alert_pipe_parent.send("request_progress")
        return self.progress_pipe_parent.recv()

    def provide_feedback(self, feedback: int) -> bool:
        self.alert_pipe_parent.send("feedback")
        self.feedback_pipe_parent.send(feedback)
        return self.feedback_pipe_parent.recv()

    def flush(self) -> None:
        self.alert_pipe_parent.send("flush")

    def is_abandoned(self) -> bool:
        # TODO
        return False

    def get_pareto_front(self) -> pd.DataFrame:
        self.alert_pipe_parent.send("pareto")
        return self.pareto_pipe_parent.recv()
