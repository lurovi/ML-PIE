import random
import statistics
import threading
import time

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
from nsgp.sampling.RandomChooserOnlineFactory import RandomChooserOnlineFactory
from nsgp.structure.TreeStructure import TreeStructure
from threads.OptimizationThread import OptimizationThread
from util.PicklePersist import PicklePersist

import numpy as np
import pandas as pd

from threads.MlPieRun import MlPieRun

RESULTS_FOLDER = 'C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\results\\'

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

hardcoded_results_size = pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\static\\size.csv")
hardcoded_results_phi = pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\static\\phi.csv")
hardcoded_results_feynman = pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\static\\feynman.csv")

# TODO move this to a neater setup
ongoing_runs = {}
ongoing_surveys = {}

# settings
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\exps\\benchmark\\windspeed.pbz2"

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
    pair_chooser = RandomChooserOnlineFactory()
    interpretability_estimate_updater = InterpretabilityEstimateUpdater(individuals=population_storage, mutex=mutex,
                                                                        interpretability_estimator=interpretability_estimator,
                                                                        encoder=tree_encoder, pair_chooser=pair_chooser)

    ml_pie_run = MlPieRun(run_id=run_id,
                          optimization_thread=optimization_thread,
                          interpretability_estimate_updater=interpretability_estimate_updater,
                          path=app.config['RESULTS_FOLDER'])
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
        run_completed(run_id)
        return {'over': 'true'}
    return dictionary


@app.route("/getProgress", methods=['GET'])
def get_progress():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    run_id = request.headers['x-access-tokens']
    if run_id not in ongoing_runs:
        abort(404)
    progress = ongoing_runs[run_id].request_progress()
    if progress >= 100:
        run_completed(run_id)
    return {'progress': progress}


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
        run_completed(run_id)
        return {
            'outcome': 'successful',
            'over': 'true'
        }


@app.route("/restart")
def restart():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    old_run_id = request.headers['x-access-tokens']
    try:
        file = open(app.config['RESULTS_FOLDER'] + 'reset.txt', 'a')
        file.write(old_run_id + '\n')
        file.close()
    except IOError:
        pass
    return start_run()


@app.route("/getSurveyData")
def get_survey_data():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    run_id = request.headers['x-access-tokens']
    if run_id not in ongoing_surveys:
        abort(404)
    comparisons = []
    for idx, row in ongoing_surveys[run_id].iterrows():
        dictionary = {
            'type': row.other_type,
            'pie_latex': row.latex_tree,
            'other_latex': row.other_latex
        }
        comparisons.append(dictionary)
    return {'comparisons': comparisons}


@app.route("/answerSurvey", methods=['POST'])
def answer_survey():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    run_id = request.headers['x-access-tokens']
    if run_id not in ongoing_surveys:
        abort(404)
    survey_data = ongoing_surveys[run_id]
    preferences = [request.json["size"], request.json["phi"], request.json["feynman"]]
    survey_data['preference'] = preferences
    survey_data.to_csv(path_or_buf=app.config['RESULTS_FOLDER'] + "survey-" + run_id + ".csv")
    return {'outcome': 'successful'}


def run_completed(run_id: str):
    if run_id not in ongoing_runs or run_id in ongoing_surveys:
        return
    results = ongoing_runs[run_id].get_pareto_front()
    try:
        del ongoing_runs[run_id]
    except KeyError:
        return

    accuracies = list(results['accuracy'])
    median_accuracy = statistics.median(accuracies)
    distances_from_median = list(map(lambda a: abs(median_accuracy - a), accuracies))
    max_distance = max(distances_from_median)
    sampling_weights = list(map(lambda d: (max_distance - d) + 1, distances_from_median))

    chosen_indexes = random.choices(range(len(results)), sampling_weights, k=3)
    chosen_models = results.loc[chosen_indexes, :].reset_index().drop(columns=['index'])

    target_accuracies = chosen_models['accuracy'].tolist()

    size_model = find_closest_model(target_accuracies[0], hardcoded_results_size)
    phi_model = find_closest_model(target_accuracies[1], hardcoded_results_phi)
    feynman_model = find_closest_model(target_accuracies[2], hardcoded_results_feynman)

    static_models = pd.concat([size_model, phi_model, feynman_model], ignore_index=True).rename(lambda c: "other_" + c,
                                                                                                axis='columns')
    ongoing_surveys[run_id] = pd.concat([chosen_models, static_models], axis=1)


def find_closest_model(target_accuracy, dataframe):
    accuracies = dataframe['accuracy'].tolist()
    accuracy_distances = list(map(lambda d: abs(d - target_accuracy), accuracies))

    min_dist = min(accuracy_distances)
    row_id = accuracy_distances.index(min_dist)
    return dataframe.iloc[[row_id]]


def runs_cleanup():
    for run_id in ongoing_runs:
        if ongoing_runs[run_id].is_abandoned():
            try:
                del ongoing_runs[run_id]
            except KeyError:
                pass
