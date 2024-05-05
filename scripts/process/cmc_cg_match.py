"""
Script to match cmc and cg id
"""

import json

import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH
from environ.fetch.coingecko import CoinGecko

# Load the panel data
dfp = pd.read_csv(PROCESSED_DATA_PATH / "panel_with_macro_cate_google.csv")
p_list = dfp["id"].unique().tolist()
cg = CoinGecko()
cg_list = cg.coins_list()
df_cg = pd.concat([pd.json_normalize(_) for _ in cg_list])
df_cg = df_cg[df_cg["id"].isin(p_list)]
df_cg.rename(columns={"id": "cg_id"}, inplace=True)

df_lst = []

# Save the matched data
for list_name in [
    "active", 
    "inactive",
    "untracked"
]:
    with open(
        DATA_PATH / "cmc" / "id" / f"cmc_id_{list_name}.json", "r", encoding="utf-8"
    ) as f:
        cmc_id_lst = json.load(f)

    df_cmc = pd.concat([pd.json_normalize(_) for _ in cmc_id_lst])
    df_cmc["symbol"] = df_cmc["symbol"].str.lower()
    df_cmc = df_cmc[["id", "symbol"]]
    df_cmc.rename(columns={"id": "cmc_id"}, inplace=True)

    # merge the data
    df = df_cg.merge(df_cmc, on="symbol", how="left")
    df.sort_values(["symbol","cmc_id"], ascending=True, inplace=True)
    df.drop_duplicates(subset=["symbol"], keep="first", inplace=True)

    # na and dropna
    df_cg = df_cg.loc[df_cg["cg_id"].isin(df.loc[df["cmc_id"].isna(), "cg_id"].unique().tolist())]
    df.dropna(subset=["cmc_id"], inplace=True)
    df_lst.append(df)

# merge the data
df_lst.append(df_cg)
df = pd.concat(df_lst)
df.loc[df["symbol"] == "loomold", "cmc_id"] = 2588
df.dropna(subset=["cmc_id"], inplace=True)
df["cmc_id"] = df["cmc_id"].astype(int)

top_list = df["cmc_id"].unique().tolist()

