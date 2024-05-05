"""
Script to fetch the categorical dummy for cryptocurrencies
"""

import time
import json

from tqdm import tqdm

from environ.constants import COINGECKO_API_KEY, DATA_PATH
from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()

# Fetch the list of categories
categories = [_["category_id"] for _ in cg.coins_cate_list()]

category_dict = dict()

# Fetch the list of coins in each category
for category in tqdm(categories):
    coins = [_["id"] for _ in cg.market(category, COINGECKO_API_KEY[1])]
    category_dict[category] = coins
    time.sleep(2)

# Save the dictionary
with open(DATA_PATH / "category" / "category.json", "w") as f:
    json.dump(category_dict, f, indent=4)
