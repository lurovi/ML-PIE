ongoing_runs = []


def start_optimization():
    print("Optimization starting")
    # get uuid
    # create population storage, mutex, sampler, encoder, problem (and all required elements)
    # create optimization thread
    # create interpretability estimate updater
    # store in local dictionary -> uuid - [optimization thread, interpretability estimate updater]
    # launch optimization thread
    # return uuid to user


def get_formulae():
    print("Formulae requested")
    # get uuid from session -> if empty error
    # get run from dict
    # check if run not over -> in that case remove from dict
    # get two formulae with sampler
    # send both formulae and the encoded vector (to have it back then)


def post_feedback():
    print("Feedback received")
    # get uuid from session -> if empty error
    # get run from dict
    # check if run not over -> in that case remove from dict
    # get encoded formulae (maybe use jwt) and feedback
    # feed them to the run
