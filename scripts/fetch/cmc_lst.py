"""
Script to fetch the data from coinmarketcap
"""

import json
import os
import time

from environ.constants import DATA_PATH
from environ.fetch.coinmarketcap import cmc_fetch

# check whether the data path exists
if not os.path.exists(DATA_PATH / "cmc" / "id"):
    os.makedirs(DATA_PATH / "cmc" / "id")

for listing_status in ["active", "inactive", "untracked"]:

    cmc_id_lst = []

    data_len = 5000
    start = 1
    while data_len == 5000:
        time.sleep(1)
        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/map"
        parameters = {
            "listing_status": listing_status,
            "start": start,
            "limit": "5000",
            "sort": "id",
        }
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": "",
        }
        data = cmc_fetch(url, parameters, headers)
        data_len = len(data["data"])
        cmc_id_lst += data["data"]
        start += 5000

    with open(
        DATA_PATH / "cmc" / "id" / f"cmc_id_{listing_status}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(cmc_id_lst, f, indent=4)
