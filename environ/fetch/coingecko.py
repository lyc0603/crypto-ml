"""
Script to fetch data from CoinGecko API
"""

import json
import time
import pandas as pd

import requests
from retry import retry

from environ.constants import DATA_PATH
from scripts.fetch.db import client
import pymongo

db = client["coingecko"]
collection = db["crypto"]
collection.create_index(
    [("id", pymongo.ASCENDING), ("time", pymongo.ASCENDING)], unique=True
)


class CoinGecko:
    """
    Class to fetch data from CoinGecko API
    """

    def __init__(self) -> None:
        pass

    def coins_list(self) -> list[dict[str, str]]:
        """
        Method to fetch the list of coins from CoinGecko API
        """
        url = "https://api.coingecko.com/api/v3/coins/list"
        response = requests.get(url, timeout=60)
        return response.json()

    def coins_cate_list(self) -> list[dict[str, str]]:
        """
        Method to fetch the list of coins categories from CoinGecko API
        """
        url = "https://api.coingecko.com/api/v3/coins/categories/list"
        response = requests.get(url, timeout=60)
        return response.json()

    def coins_cate(self, category: str) -> list[dict[str, str]]:
        """
        Method to fetch the list of coins from a category from CoinGecko API
        """
        url = f"https://api.coingecko.com/api/v3/coins/categories/{category}"
        response = requests.get(url, timeout=60)
        return response.json()

    def market(self, category) -> list[dict[str, str]]:
        """
        Method to get the market data from CoinGecko API
        """

        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category={category}"
        response = requests.get(url, timeout=60)
        return response.json()

    @retry(delay=1, backoff=2, tries=3)
    def market_data(self, coin_id: str, api_key: str) -> None:
        """
        Method to fetch the market data of a coin from CoinGecko API
        """
        url = (
            f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            + "/market_chart?vs_currency=usd&days=365"
            + f"&x_cg_demo_api_key={api_key}"
        )
        try:
            response = requests.get(url, timeout=60).json()
            df_crypto = {
                "id": [],
                "price": [],
                "mcap": [],
                "vol": [],
                "timestamp": [],
            }
            for dict_name, lst_name in {
                "price": "prices",
                "mcap": "market_caps",
                "vol": "total_volumes",
            }.items():
                df_crypto[dict_name] = df_crypto[dict_name] + [
                    _[1] for _ in response[lst_name]
                ]

            df_crypto["timestamp"] = df_crypto["timestamp"] + [
                _[0] for _ in response["prices"]
            ]
            df_crypto["id"] = df_crypto["id"] + [coin_id] * len(
                response["prices"]
            )
            df_crypto = pd.DataFrame(df_crypto)

            # convert the timestamp to datetime and remove the current timestamp
            df_crypto = df_crypto[df_crypto["timestamp"] != 0]
            df_crypto.sort_values(["id", "timestamp"], ascending=True, inplace=True)
            df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"], unit="ms")
            df_crypto["date"] = df_crypto["date"].dt.strftime("%Y-%m-%d")
            df_crypto.drop_duplicates(subset=["id", "date"], inplace=True, keep="first")

            for _, row in df_crypto.iterrows():
                data_dict = {
                    "id": row["id"],
                    "time": row["date"],
                    "price": row["price"],
                    "mcap": row["mcap"],
                    "vol": row["vol"],
                }
                try:
                    collection.insert_one(data_dict)
                except:
                    continue

            # # save as json
            # with open(DATA_PATH / "coingecko" / f"{coin_id}.json", "w") as f:
            #     json.dump(response.json(), f)

        except Exception as e:  # pylint: disable=broad-excepts
            print(f"Error:{coin_id}, {e}")

        time.sleep(2)
