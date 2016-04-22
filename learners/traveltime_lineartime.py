from sklearn import linear_model
import json
import pickle
import numpy as np
import pandas as pd
import numpy as np
import datatables.traveltime

def frac_day(time):
    """
    Convert time to fraction of a day (0.0 to 1.0)
    Can also pass this function a datetime object
    """
    return time.hour*(1./24) + time.minute*(1./(24*60)) + time.second*(1./(24*60*60))

def write_model(regr, model_file):
    """
    Write linear model to file
    regr -- trained sklearn.linear_model
    output_file -- file
    """
    model_params = {
        'coef': list(regr.coef_),
        'intercept': regr.intercept_
    }
    model_str = json.dumps(model_params)
    with open(model_file, 'w') as out_f:
        out_f.write(model_str)

def load_model(model_file):
    """
    Load linear model from file
    model_file -- file
    returns -- trained sklearn.linear_model
    """
    with open(model_file, 'r') as model_f:
        model_str = model_f.read()
    model_params = json.loads(model_str)
    regr = linear_model.LinearRegression()
    regr.coef_ = np.array(model_params['coef'])
    regr.intercept_ = model_params['intercept']
    return regr

def train(train_data_file, model_file):
    data = datatables.traveltime.read_xs(train_data_file)
    data['frac'] = [frac_day(t) for t in data['t']]

    # Extract Features
    # We create the features $time^1$, $time^2$, ... in order to allow the regression algorithm to find polynomial fits.

    # Turn list into a n*1 design matrix. At this stage, we only have a single feature in each row.
    t = data['frac'].values[:, np.newaxis]
    # Add x^2 as feature to allow quadratic regression
    xs = np.hstack([t, t**2, t**3, t**4, t**5, t**6, t**7, t**8])
    y = data['y'].values # travel times

    regr = linear_model.LinearRegression()
    regr.fit(xs, y)
    write_model(regr, model_file)

def predict(model_file, test_xs_file, output_file):
    regr = load_model(model_file)
    data = datatables.traveltime.read_xs(test_xs_file)
    data['frac'] = [frac_day(t) for t in data['t']]
    # Turn list into a n*1 design matrix. At this stage, we only have a single feature in each row.
    t = data['frac'].values[:, np.newaxis]
    # Add x^2 as feature to allow quadratic regression
    xs = np.hstack([t, t**2, t**3, t**4, t**5, t**6, t**7, t**8])

    y_pred = regr.predict(xs)
    data['pred'] = y_pred
    datatables.traveltime.write_pred(data, output_file)
