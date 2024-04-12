"""
Script to detect the database
"""

from scripts.fetch.db import client

def max_time(db_name: str, collection_name: str) -> str:
    """
    Function to get the max time of crypto
    """
    db = client[db_name]
    collection = db[collection_name]
    max_time = collection.find_one(sort=[("time", -1)])["time"]
    return max_time

# get the max time of crypto
max_time_crypto = max_time("coingecko", "crypto")

# get the max time of weekly
max_time_weekly = max_time("crypto_ml", "weekly")

# get the max time of features
max_time_feature = max_time("crypto_ml", "features")