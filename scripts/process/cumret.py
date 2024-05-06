"""
Script to calculate the cumulative return
"""

import pandas as pd
from tqdm import tqdm

from environ.constants import ML_METHOD, PROCESSED_DATA_PATH
# from scripts.process.capm import mkt
from scripts.process.portfolio import portfolio_dict

crypto_lst = pd.read_csv(PROCESSED_DATA_PATH / "crypto_lst_with_mcap.csv")
crypto_lst["time"] = pd.to_datetime(crypto_lst["time"])

cumret_plot_dict = {}

for ml in tqdm(ML_METHOD):
    cumret_plot_dict[ml] = {}
    # calculate the cumulative return for the quantile 5 and 1
    df = portfolio_dict[ml].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.loc[(df["quantile"] == 5)|(df["quantile"] == 1)].copy()
    df_q = df.groupby(["time", "quantile"])["ret_w"].mean().reset_index()
    df_q = df_q.pivot(index="time", columns="quantile", values="ret_w")
    df_q["l"] = (1 + df_q[5]).cumprod() - 1
    df_q["ls"] = (1 + df_q[5] - df_q[1]).cumprod() - 1
    cumret_plot_dict[ml] = df_q

# equal weight
dfi = df.copy()
df_q = df.groupby(["time"])["ret_w"].mean()
df_q["eq"] = (1 + df_q).cumprod() - 1
cumret_plot_dict["eq"] = df_q

# mcap weight
dfm = df.copy()
dfm = dfm.merge(crypto_lst, on=["id", "time"], how="left")
dfm["market_caps"] = dfm.groupby("time")["market_caps"].transform(lambda x: x / x.sum())
dfm["mcap"] = dfm["market_caps"] * dfm["ret_w"]
df_q = dfm.groupby(["time"])["mcap"].sum()
df_q["mcap"] = (1 + df_q).cumprod() - 1
cumret_plot_dict["mcap"] = df_q

