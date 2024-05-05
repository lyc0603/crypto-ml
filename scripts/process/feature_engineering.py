"""
Script to perform feature engineering on the data
"""

import re

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from environ.utils.db import load_db
from scripts.process.capm import rfm

pandarallel.initialize(progress_bar=True, nb_workers=40)

df = load_db(db_name="crypto_ml", collection_name="features")
df["time"] = pd.to_datetime(df["time"])


# winsorize the data
def winsor(x: pd.Series, cut=[0.01, 0.99]) -> pd.Series:
    """
    Function to winsorize the data
    """
    x = x.copy()
    lb = x.quantile(cut[0])
    ub = x.quantile(cut[1])
    x[x < lb] = lb
    x[x > ub] = ub
    return x


# calculate the weekly return
df.sort_values(["id", "time"], ascending=True, inplace=True)
df["ret"] = df.groupby(["id"])["prices"].shift(-7) / df["prices"] - 1

# calculate the daily return
df["eret"] = df.groupby(["id"])["prices"].pct_change()
df = df.merge(rfm, on="time", how="left")
df["eret"] = df["eret"] - df["rf"]
df.sort_values(["id", "time"], ascending=False, inplace=True)
df["eret"] = df.groupby(["id"])["eret"].shift(1)

# calculate the rolloing 7-day cumulative return
df_lst = []
for _, df_id in tqdm(df.copy().groupby("id")):
    df_id["eret"] = df_id["eret"].rolling(7).apply(lambda x: (1 + x).prod() - 1)
    df_lst.append(df_id)

df = pd.concat(df_lst)

df["log_ret"] = np.log(df["ret"] + 1)
df["log_eret"] = np.log(df["eret"] + 1)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.drop(columns=["prc_i", "prc_j"], inplace=True)

info_vars = ["id", "year", "week", "time", "prices"]
ret_vars = ["ret", "log_ret", "eret", "log_eret"]
wret_vars = [f"{x}_w" for x in ret_vars]

factor_vars = [x for x in df.columns if re.findall("size_|vol_|volume_|mom_", x)]
wfactor_vars = [f"{x}_w" for x in factor_vars]

for var in [x for x in factor_vars + ret_vars]:
    df[f"{var}_w"] = winsor(df[var])

df = df.loc[:, info_vars + wret_vars + wfactor_vars + ["market_caps"]]
df.dropna(inplace=True)

# only keep the top 50 market cap crypto every day using parrallel apply
df = (
    df.groupby("time")
    .parallel_apply(lambda x: x.nlargest(100, "market_caps"))
    .reset_index(drop=True)
)
df.drop(columns=["market_caps"], inplace=True)

df.to_csv(PROCESSED_DATA_PATH / "gecko_panel.csv", index=False)
