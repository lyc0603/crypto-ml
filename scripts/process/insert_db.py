"""
Script to insert data into the database
"""

from multiprocessing import Pool

import pandas as pd
import pymongo
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from scripts.fetch.db import client

N_WORKERS = 36

db = client["crypto_ml"]
collection = db["weekly"]
collection.create_index(
    [("id", pymongo.ASCENDING), ("time", pymongo.ASCENDING)], unique=True
)

# load the data after merging raw data
df_crypto = pd.read_csv(PROCESSED_DATA_PATH / "gecko_weekly.csv")
df_crypto.drop(columns=["Unnamed: 0"], inplace=True)

# convert the timestamp to datetime and remove the current timestamp
data_list = []

for _, row in tqdm(df_crypto.iterrows(), total=df_crypto.shape[0]):
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