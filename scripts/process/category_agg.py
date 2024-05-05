"""
Script to process the categorical dummy for cryptocurrencies
"""

import json
import warnings

import pandas as pd
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")

with open(DATA_PATH / "category" / "category.json", "r", encoding="utf-8") as f:
    category_dict = json.load(f)

df_cate = pd.DataFrame()

crypto_set = set()

for key in category_dict.keys():
    for value in category_dict[key]:
        crypto_set.add(value)

crypto_set = list(crypto_set)
df_cate["id"] = crypto_set

for key in tqdm(category_dict.keys()):
    for value in category_dict[key]:
        df_cate.loc[df_cate["id"] == value, key] = 1

df_cate = df_cate.fillna(0)

df_cate.set_index("id").T.to_csv(PROCESSED_DATA_PATH / "category.csv")


