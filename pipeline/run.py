from ruffus import *
import pandas as pd
import numpy as np
import importlib
import traceback
import math
import datatables.prediction
from util.util import path_to_pypath
import pipeline.task_traveltime

# taskX_learnerY.py, taskX.task.train -> taskX_learnerY.model
# Register tasks as dependencies here
@follows(pipeline.task_traveltime.split_task_traveltime)
@transform('learners/*.py',
            regex(r'^learners/([^_]*)_([^_]*).py$'),
            add_inputs(r'data/\1.task.train'),
            r'data/\1_\2.model')
def train_model(input_files, model_param_file):
    learner_pyfile, data_train = input_files
    learner_pypath = path_to_pypath(learner_pyfile)

    # Create empty model_param_file.
    # Will be overwritten by learner (or left empty if learner fails)
    with open(model_param_file, 'w'):
        pass

    try:
        learner = importlib.import_module(learner_pypath)
        learner.train(data_train, model_param_file)
    except Exception as e:
        print ('Failed to find/call train function in {0}. Exception: {1}'.format(learner_pyfile, e))
        traceback.print_exc()

# taskX_learnerY.py, taskX_learnerY.model -> taskX_learnerY.pred
@follows(train_model)
@transform('learners/*.py',
           regex(r'^learners/([^_]*)_([^_]*).py$'),
           add_inputs([r'data/\1.task.test.xs', r'data/\1_\2.model']),
           r'data/\1_\2.pred')
def make_prediction(input_files, prediction_file):
    learner_pyfile, extras = input_files
    data_test_xs, model_param_file = extras
    learner_pypath = path_to_pypath(learner_pyfile)

    # Create empty prediction_file.
    # Will be overwritten by learner (or left empty if learner fails)
    with open(prediction_file, 'w'):
        pass

    try:
        learner = importlib.import_module(learner_pypath)
        learner.predict(model_param_file, data_test_xs, prediction_file)
    except Exception as e:
        print ('Failed to find/call predict function in {0}. Exception: {1}'.format(learner_pyfile, e))
        traceback.print_exc()

# taskX_learnerY.pred, taskX.test -> taskX_learnerY.errors
@transform(make_prediction,
           regex(r'^data/([^_]*)_([^_]*)\.pred$'),
           add_inputs(r'data/\1.task.test'),
           r'data/\1_\2.errors')
def compare_prediction(input_files, output_file):
    pred_file, actual_file = input_files
    actual = datatables.prediction.read_actual(actual_file)

    try:
        pred = datatables.prediction.read_pred(pred_file)
        # Don't trust format of pred. Filter to just the fields that we expect to exist.
        pred = pred[['id', 'pred']]
        # TODO: Ensure that there are no duplicate rows in pred file, and that join is 1:1
        data = pd.merge(actual, pred, on='id', how='left')
        data['error'] = data['y'] - data['pred']
    except Exception as e:
        print ('Could not parse predictions {0}. Exception: {1}'.format(pred_file, e))
        traceback.print_exc()
        data = actual
        data['pred'] = float('nan')
        data['error'] = data['y'] - data['pred']

    datatables.prediction.write_cmp(data, output_file)

@transform(compare_prediction,
           suffix('.errors'),
           '.results')
def evaluate_prediction(input_file, output_file):
    data = datatables.prediction.read_cmp(input_file)

    # TODO: ensure data is a real number (complex numbers could be used to cheat)
    rms_error = math.sqrt(sum(data['error']**2) / len(data))

    with open(output_file, "w") as out_f:
        out_f.write(str(rms_error))

# cX_mY.res -> cX.ladder
@collate(evaluate_prediction,
         regex(r"data/([^_]*)_[^_]*\.results"),
         r"data/\1.ladder")
def comp_ladder(infiles, summary_file):
    errors = []

    for infile in infiles:
        with open(infile) as in_f:
            error = float(in_f.read())
            errors.append((infile, error))

    summary_data = pd.DataFrame(errors, columns=["model", "score"])
    # low scores (lowest error) at the top
    summary_data = summary_data.sort_values(by="score", ascending=True)
    summary_data.to_csv(summary_file, index=False)
