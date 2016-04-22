import pandas as pd

def read_actual(csv_file):
    dataframe = pd.read_csv(csv_file)
    assert 'id' in dataframe, "id not present"
    assert 'y' in dataframe, "y not present"
    return dataframe

def read_pred(csv_file):
    dataframe = pd.read_csv(csv_file)
    assert 'id' in dataframe, "id not present"
    assert 'pred' in dataframe, "pred not present"
    return dataframe

def write_cmp(dataframe, csv_file):
    assert 'id' in dataframe, "id not present"
    assert 'y' in dataframe, "y not present"
    assert 'pred' in dataframe, "pred not present"
    assert 'error' in dataframe, "error not present"
    return dataframe.to_csv(csv_file, index=False)

def read_cmp(csv_file):
    dataframe = pd.read_csv(csv_file)
    assert 'id' in dataframe, "id not present"
    assert 'y' in dataframe, "y not present"
    assert 'pred' in dataframe, "pred not present"
    assert 'error' in dataframe, "error not present"
    return dataframe
