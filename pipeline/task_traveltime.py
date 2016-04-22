from ruffus import *
import pandas as pd
import numpy as np
import datetime
import datatables.traveltime
import pipeline.data_merged

# Merge all dates into a single file
# X.merged -> traveltime.task
@merge(pipeline.data_merged.merge_data,
       'data/traveltime.task')
def task_traveltime(input_files, output_file):
    frames = []
    for input_file in sorted(input_files):
        frame = datatables.traveltime.read(input_file)
        frames.append(frame)
    all_data = pd.concat(frames, ignore_index=True)
    # Renaming the field to be predicted (travel time) as 'y', and assigning an id
    # makes it easier to reuse the same evaluation code for multiple tasks.
    all_data = all_data.rename(columns = {'travel time':'y'})
    all_data.insert(0, 'id', np.arange(all_data.shape[0])) # label ids 0,1,...
    datatables.traveltime.write(all_data, output_file)

# Prepare train and test data sets for travel time task
# traveltime.task -> traveltime.task.train, traveltime.task.test, traveltime.task.test.xs
@split(task_traveltime,
       ['data/traveltime.task.train', 'data/traveltime.task.test', 'data/traveltime.task.test.xs'])
def split_task_traveltime(input_file, output_files):
    all_data = datatables.traveltime.read(input_file)
    train_out, test_out, test_xs = output_files

    # split the training set into test/train at thresh date
    thresh = datetime.datetime(2015,8,26)
    low_dates = all_data[all_data['t'] < thresh]
    high_dates = all_data[all_data['t'] >= thresh]

    datatables.traveltime.write(low_dates, train_out)
    datatables.traveltime.write(high_dates, test_out)
    # hide travel time column to be predicted
    high_dates_xs = high_dates.drop('y', 1)
    datatables.traveltime.write_xs(high_dates_xs, test_xs)
