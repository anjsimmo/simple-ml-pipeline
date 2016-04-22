import json
import pickle
import numpy as np
import pandas as pd
import numpy as np
import datatables.traveltime

def write_model(baserate, model_file):
    """
    Write model to file
    baserate -- average travel time
    output_file -- file
    """
    model_params = {
        'baserate': baserate
    }
    model_str = json.dumps(model_params)
    with open(model_file, 'w') as out_f:
        out_f.write(model_str)

def load_model(model_file):
    """
    Load linear model from file
    model_file -- file
    returns -- baserate
    """
    with open(model_file, 'r') as model_f:
        model_str = model_f.read()
    model_params = json.loads(model_str)
    return model_params['baserate']

def train(train_data_file, model_file):
    data = datatables.traveltime.read_xs(train_data_file)
    y = data['y'].values # travel times
    # use mean value as baserate prediction
    baserate = np.mean(y)
    write_model(baserate, model_file)

def predict(model_file, test_xs_file, output_file):
    baserate = load_model(model_file)
    data = datatables.traveltime.read_xs(test_xs_file)
    num_rows = data.shape[0]
    # predict constant baserate for every row
    y_pred = np.full(num_rows, baserate)
    data['pred'] = y_pred
    datatables.traveltime.write_pred(data, output_file)
