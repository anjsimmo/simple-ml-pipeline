import subprocess
import datetime
from ruffus import *
import pandas as pd
import re
import urllib.request

@originate("data/VSDATA_20150819.csv")
def vs19(output_file):
    # Download file from AARNet Cloudstor OwnCloud Service
    url = "https://cloudstor.aarnet.edu.au/plus/index.php/s/SlTMKzq9OKOaWQr/download?path=%2Fvicroads_opendata&files=VSDATA_20150819.csv"
    print("Downloading {0} from {1}".format(output_file, url))
    urllib.request.urlretrieve(url, output_file)

@originate("data/VSDATA_20150826.csv")
def vs26(output_file):
    url = "https://cloudstor.aarnet.edu.au/plus/index.php/s/SlTMKzq9OKOaWQr/download?path=%2Fvicroads_opendata&files=VSDATA_20150826.csv"
    print("Downloading {0} from {1}".format(output_file, url))
    urllib.request.urlretrieve(url, output_file)

# X.csv -> X.volume
@transform([vs19, vs26],
           suffix(".csv"),
           ".volume")
def import_vs(input_file, output_file):
    # Import data
    f = pd.read_csv(input_file)

    # Filter to site 2433 (mid-way along segment of Princes freeway monitored by bluetooth detector sites).
    # Detectors 4-6 are in the outbound/westbound lanes.
    vols = f[(f["NB_SCATS_SITE"] == 2433) & f["NB_DETECTOR"].between(4,6)]

    # Extract date from CSV data
    start_date = vols["QT_INTERVAL_COUNT"].iloc[0]
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d 00:00:00')
    date_range = pd.date_range(start_datetime, periods=96, freq='15T')

    # Transpose table. Label by time rather than interval. Use detector number as headers.
    dets = vols.T
    dets.columns = dets.loc["NB_DETECTOR"].values
    dets = dets.loc['V00':'V95']
    dets.index=date_range

    # Extract just detector 6 (the rightmost lane)
    d6 = dets[6]

    # Volume Site 2433 Detector 6 (rightmost lane) along Princes Highway (Outbound/Westbound).
    d6.to_csv(output_file)
