import pandas as pd

"""
Manages reading and writing traveltime file.
Contains specialized variants for train and test datasets, as some variants require extra or hidden columns.
"""

def write(dataframe, csv_table):
    return dataframe.to_csv(csv_table, index=False)

def read(csv_file):
    # Second column contains timestamp (first column is id)
    return pd.read_csv(csv_file, parse_dates=[1])

def write_xs(dataframe, csv_table):
    # Only difference is a missing 'y' column.
    return write(dataframe, csv_table)

def read_xs(csv_file):
    return read(csv_file)

def write_pred(dataframe, csv_file):
    # Only difference is a missing 'y' column, and an extra 'pred' column
    return write(dataframe, csv_file)
