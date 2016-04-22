from ruffus import *
import pandas as pd
from util.util import file_by_type
import datatables.traveltime
import pipeline.data_bt
import pipeline.data_vs

# BLUETH_YYYYMMDD.traveltime, VSDATA_YYYYMMDD.volume -> YYYYMMDD.merged
@collate([pipeline.data_bt.import_bt, pipeline.data_vs.import_vs],
         regex(r"^data/(BLUETH|VSDATA)_(\d{8})\.(traveltime|volume)$"),
         r"data/\2.merged")
def merge_data(infiles, mergefile):
    assert (len(infiles) == 2), "Expected exactly 2 files (BLUETH_... and VSDATA_...) to merge"
    bt_f = file_by_type(infiles, '.traveltime') # 'BLUETH_...'
    vs_f = file_by_type(infiles, '.volume') # 'VSDATA_...'
    bt = pd.read_csv(bt_f, header=None, names=['t', 'travel time'])
    vs = pd.read_csv(vs_f, header=None, names=['t', 'volume'])
    data = pd.merge(vs, bt, on='t', how='left')
    datatables.traveltime.write(data, mergefile)
