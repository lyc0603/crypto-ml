"""
Script to calculate the cumulative return
"""

import pandas as pd

from environ.constants import ML_METHOD
from scripts.process.portfolio import portfolio_dict
from scripts.process.capm import mkt


# calculate the 7-day forward return for mkt
mkt_w = []
mkt = mkt.reset_index().dropna()
for time in mkt["time"].unique():
    mkt_period = mkt.loc[
        (mkt["time"] >= time) & (mkt["time"] <= time + pd.DateOffset(days=6))
    ].copy()
    if len(mkt_period) < 7:
        continue
    # calculate the cumprod of the return
    mkt_w.append(
        {"time": time, "cmkt_w": (1 + mkt_period["cmkt"]).cumprod().iloc[-1] - 1}
    )
mkt_w = pd.DataFrame(mkt_w)

cumret_plot_dict = {}

for ml in ML_METHOD:
    df = portfolio_dict[ml].copy()
    df["time"] = pd.to_datetime(df["time"])
    # groupby time and quantile to calculate the equal-weighted return
    df_q = df.groupby(["time", "quantile"])["ret_w"].mean().reset_index()
    df = df.merge(mkt_w, on="time", how="left")
    df["ret_w"] = df["ret_w"] - df["cmkt_w"]
    df_q["cum_ret"] = (1 + df_q["ret_w"]).groupby(df_q["quantile"]).cumprod()

    cumret_plot_dict[ml] = df_q

ls_plot_dict = {}

for ml in ML_METHOD:
    df = portfolio_dict[ml].copy()
    df["time"] = pd.to_datetime(df["time"])
    df_ls = []
    for time in df["time"].unique():
        df_time = df.loc[df["time"] == time].copy().reset_index(drop=True)
        df_ls.append(
            {
                "time": time,
                "ls": df_time.loc[df_time["quantile"] == 5, "ret_w"].mean()
                - df_time.loc[df_time["quantile"] == 1, "ret_w"].mean(),
            }
        )
    df_ls = pd.DataFrame(df_ls)
    # calculate the cumulative return
    df_ls["cum_ret"] = (1 + df_ls["ls"]).cumprod()
    ls_plot_dict[ml] = df_ls
