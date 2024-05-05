"""
Function to convert the data
"""

from multiprocessing import Pool

import pandas as pd
import pymongo
from tqdm import tqdm

from scripts.fetch.db import client

N_WORKERS = 36


def insert_db(
    df: pd.DataFrame,
    db_name: str = "crypto_ml",
    collection_name: str = "weekly",
    index: list = ["id", "time"],
) -> None:
    """
    Function to insert pd.DataFrame into the mongodb database
    """
    db = client[db_name]
    collection = db[collection_name]
    collection.create_index(
        [(index_name, pymongo.ASCENDING) for index_name in index], unique=True
    )

    # convert the timestamp to datetime and remove the current timestamp
    data_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        data_dict = row.to_dict()
        data_list.append(data_dict)

    with Pool(N_WORKERS) as p:
        list(tqdm(p.imap(collection.insert_one, data_list), total=len(data_list)))

def load_db(
    db_name: str = "crypto_ml",
    collection_name: str = "weekly",
) -> pd.DataFrame:
    """
    Function to load the data from the mongodb database
    """
    db = client[db_name]
    collection = db[collection_name]

    df_crypto = []

    # Iterate over each document in the collection
    for document in tqdm(collection.find()):
        df_crypto.append(document)

    # convert the data to dataframe
    return pd.DataFrame(df_crypto)