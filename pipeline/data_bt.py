import subprocess
import datetime
from ruffus import *
import pandas as pd
import re
import urllib.request

@originate("data/BLUETH_20150819.BT")
def bt19(output_file):
    # Download file from AARNet Cloudstor OwnCloud Service
    url = "https://cloudstor.aarnet.edu.au/plus/index.php/s/SlTMKzq9OKOaWQr/download?path=%2Fvicroads_opendata&files=BLUETH_20150819.BT"
    print("Downloading {0} from {1}".format(output_file, url))
    urllib.request.urlretrieve(url, output_file)

@originate("data/BLUETH_20150826.BT")
def bt26(output_file):
    url = "https://cloudstor.aarnet.edu.au/plus/index.php/s/SlTMKzq9OKOaWQr/download?path=%2Fvicroads_opendata&files=BLUETH_20150826.BT"
    print("Downloading {0} from {1}".format(output_file, url))
    urllib.request.urlretrieve(url, output_file)

# X.BT -> X.filtered.BT
@transform([bt19, bt26],
           suffix(".BT"),
           ".filtered.BT")
def filter_bt(input_file, output_file):
    # filter to just sites 2425 and 2409
    # grep is a lot faster than processing the file in Python
    with open(output_file, 'w') as outfile:
        subprocess.call(['grep', '-E', '^(2425|2409),', input_file], stdout=outfile)

def segments(df):
    """
    Convert ordered table of visited sites into segments between adjacent nodes.
    dataframe -- site, time, bluetooth_id
    """
    results = []
    last_row = None
    for index, row in df.iterrows():
        if last_row is not None and row["Site"] != last_row["Site"]:
            segment = (last_row["Anonymized Bluetooth ID"],
                       last_row["Site"],
                       row["Site"],
                       last_row["Unix Time"],
                       row["Unix Time"])
            results.append(segment)
        last_row = row
    return results

def parse_date(unix_time):
    d_utc = datetime.datetime.utcfromtimestamp(unix_time)
    # Unix servers *should* have their system clock set to UTC.
    # So we theoretically, we need to convert from UTC to AEST (localtime).
    # However, VicRoads seems to have set their operating system clock to AEST.
    # The easiest way to deal with this, is to treat all datetimes as naive (ignore timezone).
    # TLDR; VicRoads didn't handle timezones correctly. We need to copy their error for consistency.
    d_local = d_utc # Naive datetime. It's already shifted to AEST (but shouldn't be)
    return d_local

# X.filtered.BT -> X.traveltime
@transform(filter_bt,
           suffix(".filtered.BT"),
           ".traveltime")
def import_bt(input_file, output_file):
    # Load into Pandas Data Table
    f = pd.read_csv(input_file, header=None, names=['Site', 'Unix Time', 'Anonymized Bluetooth ID'])
    f_sorted = f.sort_values(by=['Anonymized Bluetooth ID', 'Unix Time'])
    f_groups = f_sorted.groupby(['Anonymized Bluetooth ID'])

    results = []
    for bt_id, data in f_groups:
        for segment in segments(data):
            results.append(segment)

    all_segments = pd.DataFrame(results,
                                columns=('Anonymized Bluetooth ID', 'Site A', 'Site B', 'Time A', 'Time B'))

    inbound = all_segments[all_segments["Site A"] == 2409]
    inbound = inbound.copy()
    travel_time = inbound["Time B"] - inbound["Time A"]
    inbound["Travel Time"] = travel_time

    # Filter extreme travel times
    inbound = inbound[inbound["Travel Time"] <= 1800]

    ts = pd.Series(list(inbound["Travel Time"]),
                   index=list([parse_date(t) for t in inbound["Time A"]]))

    ts_resampled = ts.resample('15Min', how='median')

    # extract collection date from filename
    p = re.compile(r"data/BLUETH_(?P<date>\d{8})\.filtered.BT")
    m = p.match(input_file)
    date_str = m.group('date')
    start_datetime = datetime.datetime.strptime(date_str, '%Y%m%d')

    # Index over entire day, even if some times are missing. Last 15 minutes usualy not present.
    rng = pd.date_range(start_datetime, periods=24*4, freq='15Min')
    ts_resampled = pd.Series(ts_resampled, index=rng)

    # Fill in missing values
    ts_resampled = ts_resampled.fillna(method='pad')

    # Travel time from site 2409 (Chapel St) to 2425 (Warrigal Rd) along Princes Highway (Outbound/Westbound).
    ts_resampled.to_csv(output_file)
