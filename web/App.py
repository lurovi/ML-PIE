import random
import statistics
import threading

import torch
import uuid
from flask import Flask, request, render_template, abort
from pymoo.algorithms.moo.nsga2 import NSGA2
from torch import nn

from deeplearn.model.MLPNet import MLPNet
from deeplearn.trainer.OnlineTwoPointsCompareTrainer import OnlineTwoPointsCompareTrainer
from deeplearn.trainer.TwoPointsCompareTrainerFactory import TwoPointsCompareTrainerFactory
from exps.DatasetGenerator import DatasetGenerator
from exps.groundtruth.MathElasticModelComputer import MathElasticModelComputer
from genepro import node_impl
from nsgp.callback.PopulationAccumulator import PopulationAccumulator
from nsgp.encoder.CountsEncoder import CountsEncoder
from nsgp.interpretability.InterpretabilityEstimateUpdater import InterpretabilityEstimateUpdater
from nsgp.operator.TreeSetting import TreeSetting
from nsgp.problem.RegressionProblemWithNeuralEstimate import RegressionProblemWithNeuralEstimate
from nsgp.sampling.UncertaintyChooserOnlineFactory import UncertaintyChooserOnlineFactory
from nsgp.structure.TreeStructure import TreeStructure
from threads.OptimizationThread import OptimizationThread
from util.PicklePersist import PicklePersist

import numpy as np
import pandas as pd

from threads.MlPieRun import MlPieRun

RESULTS_FOLDER = 'C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\results\\'

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

hardcoded_results = {
    "heating": {
        "size": pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\gpresults\\heating_size.csv"),
        "phi": pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\gpresults\\heating_phi.csv"),
        "wu_phi": pd.read_csv(
            "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\gpresults\\heating_wu_phi.csv")
    },
    "boston": {
        "size": pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\gpresults\\boston_size.csv"),
        "phi": pd.read_csv("C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\gpresults\\boston_phi.csv"),
        "wu_phi": pd.read_csv(
            "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\gpresults\\boston_wu_phi.csv")
    }
}

available_problems = {
    "heating": "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\exps\\benchmark\\heating.pbz2",
    "boston": "C:\\Users\\giorg\\PycharmProjects\\ML-PIE\\exps\\benchmark\\boston.pbz2"
}

ongoing_runs = {}
ongoing_surveys = {}

run_problems_associations = {}

# settings
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tree parameters
n_features_boston, n_features_heating = 13, 8
phi = MathElasticModelComputer()
duplicates_elimination_data_boston = np.random.uniform(-5.0, 5.0, size=(5, n_features_boston))
seed = 100
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
duplicates_elimination_data_heating = np.random.uniform(-5.0, 5.0, size=(5, n_features_heating))
internal_nodes = [node_impl.Plus(), node_impl.Minus(), node_impl.Times(), node_impl.Div(),
                  node_impl.Cube(),
                  node_impl.Log(), node_impl.Max()]
normal_distribution_parameters_boston = [(0, 1), (0, 1), (0, 3), (0, 8),
                                         (0, 8),
                                         (0, 30), (0, 15)] + [(0, 0.8)] * n_features_boston + [(0, 0.5)]
normal_distribution_parameters_heating = [(0, 1), (0, 1), (0, 3), (0, 8),
                                          (0, 8),
                                          (0, 30), (0, 15)] + [(0, 0.8)] * n_features_heating + [(0, 0.5)]
structure_boston = TreeStructure(internal_nodes, n_features_boston, 5,
                                 ephemeral_func=lambda: np.random.uniform(-5.0, 5.0),
                                 normal_distribution_parameters=normal_distribution_parameters_boston)
structure_heating = TreeStructure(internal_nodes, n_features_heating, 5,
                                  ephemeral_func=lambda: np.random.uniform(-5.0, 5.0),
                                  normal_distribution_parameters=normal_distribution_parameters_heating)

print("encoder boston init...")
tree_encoder_boston = CountsEncoder(structure_boston, True, 100)
structure_boston.register_encoder(tree_encoder_boston)
print("encoder boston ready")
setting_boston = TreeSetting(structure_boston, duplicates_elimination_data_boston)
tree_sampling_boston = setting_boston.get_sampling()
tree_crossover_boston = setting_boston.get_crossover()
tree_mutation_boston = setting_boston.get_mutation()
duplicates_elimination_boston = setting_boston.get_duplicates_elimination()

print("encoder heating init...")
tree_encoder_heating = CountsEncoder(structure_heating, True, 100)
structure_heating.register_encoder(tree_encoder_heating)
print("encoder heating ready")
setting_heating = TreeSetting(structure_heating, duplicates_elimination_data_heating)
tree_sampling_heating = setting_heating.get_sampling()
tree_crossover_heating = setting_heating.get_crossover()
tree_mutation_heating = setting_heating.get_mutation()
duplicates_elimination_heating = setting_heating.get_duplicates_elimination()

train_size = 1250
validation_size = 370
test_size = 250

data_generator_boston = DatasetGenerator("boston_data_generator",
                                         structure_boston, train_size, validation_size, test_size, 101)
data_generator_boston.generate_tree_encodings(True)
data_generator_boston.generate_ground_truth([phi])
data_generator_boston.create_dataset_warm_up_from_encoding_ground_truth(20, tree_encoder_boston.get_name(), phi, 102)

data_generator_heating = DatasetGenerator("heating_data_generator",
                                          structure_heating, train_size, validation_size, test_size, 101)
data_generator_heating.generate_tree_encodings(True)
data_generator_heating.generate_ground_truth([phi])
data_generator_heating.create_dataset_warm_up_from_encoding_ground_truth(20, tree_encoder_heating.get_name(), phi, 102)


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


@app.route("/startRun/<problem>")
def start_run(problem):
    run_id = str(uuid.uuid1())
    np.random.seed(None)
    rnd_seed = np.random.randint(1, 10000)
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)

    tree_encoder = tree_encoder_boston if problem == 'boston' else tree_encoder_heating
    tree_sampling = tree_sampling_boston if problem == 'boston' else tree_sampling_heating
    tree_crossover = tree_crossover_boston if problem == 'boston' else tree_crossover_heating
    tree_mutation = tree_mutation_boston if problem == 'boston' else tree_mutation_heating
    duplicates_elimination = duplicates_elimination_boston if problem == 'boston' else duplicates_elimination_heating

    # shared parameters
    pretrainer_factory = TwoPointsCompareTrainerFactory(False, 1)
    warmup_data = data_generator_boston.get_warm_up_data(tree_encoder.get_name(), phi.get_name()) if problem == "boston" else data_generator_heating.get_warm_up_data(tree_encoder.get_name(), phi.get_name())
    mlp_net = MLPNet(nn.ReLU(), nn.Identity(), tree_encoder.size(), 1, [220, 110, 25], dropout_prob=0.25)
    interpretability_estimator = OnlineTwoPointsCompareTrainer(mlp_net, device,
                                                               warmup_trainer_factory=pretrainer_factory,
                                                               warmup_dataset=warmup_data)
    mutex = threading.Lock()
    population_storage = set()

    # optimization thread creation
    algorithm = NSGA2(pop_size=200,
                      sampling=tree_sampling,
                      crossover=tree_crossover,
                      mutation=tree_mutation,
                      eliminate_duplicates=duplicates_elimination
                      )
    # safe fallback
    if problem not in available_problems:
        problem = "boston"
    dataset = PicklePersist.decompress_pickle(available_problems.get(problem))
    regression_problem = RegressionProblemWithNeuralEstimate(dataset["training"][0], dataset["training"][1],
                                                             mutex=mutex,
                                                             tree_encoder=tree_encoder,
                                                             interpretability_estimator=interpretability_estimator
                                                             )
    termination = ('n_gen', 50)
    optimization_seed = seed
    callback = PopulationAccumulator(population_storage=population_storage)
    optimization_thread = OptimizationThread(
        optimization_algorithm=algorithm,
        problem=regression_problem,
        termination=termination,
        seed=optimization_seed,
        callback=callback,
        verbose=False,
        save_history=False
    )

    # feedback thread creation
    pair_chooser = UncertaintyChooserOnlineFactory()
    interpretability_estimate_updater = InterpretabilityEstimateUpdater(individuals=population_storage, mutex=mutex,
                                                                        interpretability_estimator=interpretability_estimator,
                                                                        encoder=tree_encoder, pair_chooser=pair_chooser)

    ml_pie_run = MlPieRun(run_id=run_id,
                          optimization_thread=optimization_thread,
                          interpretability_estimate_updater=interpretability_estimate_updater,
                          path=app.config['RESULTS_FOLDER'],
                          parameters={"problem": problem})
    ongoing_runs[run_id] = ml_pie_run
    run_problems_associations[run_id] = problem
    ml_pie_run.start()
    return {"id": run_id, "problem": problem}


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


@app.route("/reset")
def reset():
    if 'x-access-tokens' not in request.headers:
        abort(404)
    old_run_id = request.headers['x-access-tokens']
    try:
        file = open(app.config['RESULTS_FOLDER'] + 'reset.txt', 'a')
        file.write(old_run_id + '\n')
        file.close()
    except IOError:
        pass
    return {}


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
            'type': row.other_ground_truth_type,
            'pie_latex': row.latex_tree,
            'other_latex': row.other_latex_tree
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
    preferences = [request.json["size"], request.json["phi"], request.json["wu_phi"]]
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

    problem = run_problems_associations[run_id]
    try:
        del run_problems_associations[run_id]
    except KeyError:
        pass

    accuracies = list(results['accuracy'])
    median_accuracy = statistics.median(accuracies)
    distances_from_median = list(map(lambda a: abs(median_accuracy - a), accuracies))
    max_distance = max(distances_from_median)
    sampling_weights = list(map(lambda d: (max_distance - d) + 1, distances_from_median))

    chosen_indexes = random.choices(range(len(results)), sampling_weights, k=3)
    chosen_models = results.loc[chosen_indexes, :].reset_index().drop(columns=['index'])

    target_accuracies = chosen_models['accuracy'].tolist()

    size_model = find_closest_model(target_accuracies[0], hardcoded_results[problem]["size"])
    phi_model = find_closest_model(target_accuracies[1], hardcoded_results[problem]["phi"])
    wu_phi_model = find_closest_model(target_accuracies[2], hardcoded_results[problem]["wu_phi"])

    static_models = pd.concat([size_model, phi_model, wu_phi_model], ignore_index=True).rename(lambda c: "other_" + c,
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
