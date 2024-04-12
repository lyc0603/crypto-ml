"""
Script to aggregate daily data to weekly data.
"""

from multiprocessing import Pool

import pandas as pd
import pymongo
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from environ.process.clean_data import agg
from scripts.fetch.db import client
from scripts.process.detect_db import max_time_crypto, max_time_weekly

N_WORKERS = 36

db = client["crypto_ml"]
collection = db["weekly"]
collection.create_index(
    [("id", pymongo.ASCENDING), ("time", pymongo.ASCENDING)], unique=True
)

# save the daily data
df_crypto = pd.read_csv(PROCESSED_DATA_PATH / "gecko_daily.csv")

df_week = agg(
    df_crypto,
    start_time=max_time_weekly,
    end_time=max_time_crypto,
    time_col="time",
)

# convert timestamp to string
df_week["time"] = df_week["time"].astype(str)
data_list = []
for _, row in tqdm(df_week.iterrows(), total=df_week.shape[0]):
    data_dict = row.to_dict()
    data_list.append(data_dict)

def insert_data(line: dict):
    """
    Function to insert data into the database
    """
    try:
        collection.insert_one(line)
    except: # pylint: disable=broad-except
        pass

with Pool(N_WORKERS) as p:
    list(tqdm(p.imap(insert_data, data_list), total=len(data_list)))
