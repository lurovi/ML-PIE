import glob
import os
import random
import threading

import torch
import uuid
from flask import Flask, request, render_template, abort
from pymoo.algorithms.moo.nsga2 import NSGA2
from torch import nn

from deeplearn.model.MLPNet import MLPNet
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from genepro import node_impl
from nsgp.callback.PopulationAccumulator import PopulationAccumulator
from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.RegressionProblemWithNeuralEstimate import RegressionProblemWithNeuralEstimate
from nsgp.sampling.RandomChooserOnline import RandomChooserOnline
from nsgp.structure.TreeStructure import TreeStructure
from threads.OptimizationThread import OptimizationThread
from util.PicklePersist import PicklePersist

import numpy as np

from web.MlPieRun import MlPieRun

app = Flask(__name__)

# TODO move this to a neater setup
ongoing_runs = {}

# settings
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\exps\\windspeed\\wind_dataset_split.pbz2"

# tree parameters
duplicates_elimination_little_data = np.random.uniform(0.0, 1.0, size=(10, 7))
internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                  node_impl.UnaryMinus(), node_impl.Power(), node_impl.Square(), node_impl.Cube(),
                  node_impl.Sqrt(), node_impl.Exp(), node_impl.Log(), node_impl.Sin(), node_impl.Cos()]
normal_distribution_parameters = [(0, 1), (0, 1), (0, 3), (0, 8), (0, 0.5), (0, 15), (0, 5), (0, 8), (0, 20),
                                  (0, 30), (0, 30), (0, 23), (0, 23), (0, 0.8), (0, 0.8), (0, 0.8), (0, 0.8),
                                  (0, 0.8), (0, 0.8), (0, 0.8), (0, 0.5)]
structure = TreeStructure(internal_nodes, 7, 5, ephemeral_func=lambda: np.random.uniform(-5.0, 5.0),
                          normal_distribution_parameters=normal_distribution_parameters)
setting = TreeSetting(structure, duplicates_elimination_little_data)
tree_sampling = setting.get_sampling()
tree_crossover = setting.get_crossover()
tree_mutation = setting.get_mutation()
duplicates_elimination = setting.get_duplicates_elimination()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


@app.route("/survey")
def survey():
    return render_template("survey.html")


@app.route("/thanks")
def thanks():
    return render_template("thanks.html")


@app.route("/startRun")
def start_run():
    run_id = str(uuid.uuid1())

    # shared parameters
    tree_encoder = CountsEncoder(structure)
    mlp_net = MLPNet(nn.ReLU(), nn.Identity(), tree_encoder.size(), 1, [220, 110, 25])
    interpretability_estimator = OnlineTwoPointsCompareTrainer(mlp_net, device)
    mutex = threading.Lock()
    population_storage = set()

    # optimization thread creation
    algorithm = NSGA2(pop_size=20,
                      sampling=tree_sampling,
                      crossover=tree_crossover,
                      mutation=tree_mutation,
                      eliminate_duplicates=duplicates_elimination
                      )
    dataset = PicklePersist.decompress_pickle(dataset_path)
    problem = RegressionProblemWithNeuralEstimate(dataset["training"][0], dataset["training"][1], mutex=mutex,
                                                  tree_encoder=tree_encoder,
                                                  interpretability_estimator=interpretability_estimator
                                                  )
    termination = ('n_gen', 20)
    optimization_seed = seed
    callback = PopulationAccumulator(population_storage=population_storage)
    optimization_thread = OptimizationThread(
        optimization_algorithm=algorithm,
        problem=problem,
        termination=termination,
        seed=optimization_seed,
        callback=callback
    )

    # feedback thread creation
    pair_chooser = RandomChooserOnline()
    interpretability_estimate_updater = InterpretabilityEstimateUpdater(individuals=population_storage, mutex=mutex,
                                                                        interpretability_estimator=interpretability_estimator,
                                                                        encoder=tree_encoder, pair_chooser=pair_chooser)

    ml_pie_run = MlPieRun(run_id, optimization_thread, interpretability_estimate_updater)
    ongoing_runs[run_id] = ml_pie_run
    ml_pie_run.start()
    return {"id": run_id}


@app.route("/getData", methods=['GET'])
def get_data():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    run_id = request.headers['x-access-tokens']
    if run_id not in ongoing_runs:
        abort(404)
    dictionary = ongoing_runs[run_id].request_models()
    if not dictionary:
        try:
            del ongoing_runs[run_id]
        except KeyError:
            pass
        return {'over': 'true'}
    return dictionary


@app.route("/provideFeedback", methods=['POST'])
def provide_feedback():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    run_id = request.headers['x-access-tokens']
    if run_id not in ongoing_runs:
        abort(404)
    feedback_outcome = ongoing_runs[run_id].provide_feedback(int(request.json["feedback"]))
    if feedback_outcome:
        return {'outcome': 'successful'}
    else:
        try:
            del ongoing_runs[run_id]
        except KeyError:
            pass
        return {
            'outcome': 'successful',
            'over': 'true'
        }


@app.route("/restart")
def restart():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    old_run_id = request.headers['x-access-tokens']
    path = "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\results\\"
    files = glob.glob(path + "*-" + old_run_id + ".*")
    for file in files:
        try:
            os.remove(file)
        except OSError:
            pass
    return start_run()


def runs_cleanup():
    for run_id in ongoing_runs:
        if ongoing_runs[run_id].is_abandoned():
            try:
                del ongoing_runs[run_id]
            except KeyError:
                pass
