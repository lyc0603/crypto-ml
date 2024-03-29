"""
Script to fetch data from CoinGecko API
"""

import json
import time

import requests
from retry import retry

from environ.constants import DATA_PATH


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
            + "/market_chart?vs_currency=usd&days=max"
            + f"&x_cg_demo_api_key={api_key}"
        )
        try:
            response = requests.get(url, timeout=60)
            with open(DATA_PATH / "coingecko" / f"{coin_id}.json", "w") as f:
                json.dump(response.json(), f)
        except Exception as e:  # pylint: disable=broad-excepts
            print(f"Error: {e}")

        time.sleep(2)


