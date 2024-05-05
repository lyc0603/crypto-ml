"""
Fetch the data from coinmarketcap
"""
import json

import requests


def cmc_fetch(url: str, parameters: dict, headers: dict) -> dict:
    """
    Function to fetch the data from coinmarketcap
    """

    response = requests.get(url, params=parameters, headers=headers)
    print(parameters)
    return json.loads(response.text)