"""
Script to fetch the data from the CMC API
"""

import os
import time
from tqdm import tqdm

import pandas as pd

from environ.constants import DATA_PATH
from environ.fetch.coinmarketcap import cmc_fetch
from scripts.process.cmc_cg_match import top_list

# check whether the data path exists
if not os.path.exists(DATA_PATH / "cmc" / "ohlcv"):
    os.makedirs(DATA_PATH / "cmc" / "ohlcv")

# check whether the data has been fetched
fetched_list = os.listdir(DATA_PATH / "cmc" / "ohlcv")
fetched_list = [int(_.split("_")[-1].split(".")[0]) for _ in fetched_list]
top_list = [_ for _ in top_list if _ not in fetched_list]

for id in tqdm(top_list):
    try:
        time_start = "2000-01-01"
        time_end = (pd.to_datetime("today") - pd.Timedelta(
                days=2
            )).strftime("%Y-%m-%d")

        data_list = []
        n = 0
        while True:
            time.sleep(1)
            n += 1
            url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
            parameters = {
                "id": id,
                "time_period": "hourly",
                "interval": "1h",
                "count": 10000,
                "convert": "USD",   
                "time_start": time_start,
            }
            headers = {
                "Accepts": "application/json",
                "X-CMC_PRO_API_KEY": "",
            }
            data = cmc_fetch(url, parameters, headers)
            quotes = data["data"]["quotes"]
            time_start = (pd.to_datetime(quotes[-1]["time_open"]) - pd.Timedelta(
                days=1
            )).strftime("%Y-%m-%d")

            print(time_start, time_end)

            df_id = pd.concat([pd.json_normalize(_) for _ in quotes])
            data_list.append(df_id)
            if time_start >= time_end:
                break

            if n > 20:
                raise ValueError(f"Too many requests for {id}")

        df = pd.concat(data_list)
        df.drop_duplicates(inplace=True)

        df.to_csv(DATA_PATH / "cmc" / "ohlcv" / f"ohlcv_{id}.csv", index=False)
    except: # pylint: disable=broad-excepts
        continue


