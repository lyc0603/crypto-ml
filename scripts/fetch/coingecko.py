"""
Script to fetch all crypto prices from CoinGecko API
"""

import os
import glob
from multiprocessing import Pool

from tqdm import tqdm

from environ.constants import COINGECKO_API_KEY, DATA_PATH
from environ.fetch.coingecko import CoinGecko

cg = CoinGecko()
coin_list = cg.coins_list()

# os.path.exists(DATA_PATH / "coingecko") or os.makedirs(DATA_PATH / "coingecko")
# # get all the existing files names
# existing_files = glob.glob(str(DATA_PATH / "coingecko" / "*.json"))
# existing_coin_ids = [os.path.basename(file).split(".")[0] for file in existing_files]
# coin_list = [coin for coin in coin_list if coin["id"] not in existing_coin_ids]

# distribute the api key to each process
api_key_multi_list = []

for idx, coin_info in enumerate(coin_list):
    api_key_multi_list.append(
        (coin_info["id"], COINGECKO_API_KEY[idx % len(COINGECKO_API_KEY)])
    )

with Pool(processes=len(COINGECKO_API_KEY)) as p:
    list(tqdm(p.starmap(cg.market_data, api_key_multi_list), desc="Processing data"))
