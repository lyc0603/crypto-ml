"""
Script to calculate the cumulative return
"""

import pandas as pd
from tqdm import tqdm

from environ.constants import ML_METHOD
# from scripts.process.capm import mkt
from scripts.process.portfolio import portfolio_dict

# # calculate the 7-day forward return for mkt
# mkt_w = []
# mkt = mkt.reset_index().dropna()
# for time in mkt["time"].unique():
#     mkt_period = mkt.loc[
#         (mkt["time"] >= time) & (mkt["time"] <= time + pd.DateOffset(days=6))
#     ].copy()
#     if len(mkt_period) < 7:
#         continue
#     # calculate the cumprod of the return
#     mkt_w.append(
#         {"time": time, "cmkt_w": (1 + mkt_period["cmkt"]).cumprod().iloc[-1] - 1}
#     )

# mkt_w = pd.DataFrame(mkt_w)

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
