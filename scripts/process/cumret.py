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
        (mkt["time"] >= time)&(
            mkt["time"] <= time + pd.DateOffset(days=6)
        )
    ].copy()
    if len(mkt_period) < 7:
        continue
    # calculate the cumprod of the return
    mkt_w.append({
        "time": time,
        "cmkt_w": (1 + mkt_period["cmkt"]).cumprod().iloc[-1] - 1
    })
mkt_w = pd.DataFrame(mkt_w)

cumret_plot_dict = {}

for ml in ML_METHOD:
    cumret_plot_dict[ml] = {}
    for year in range(2021, 2025, 1):
        df = portfolio_dict[ml].copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.loc[df["time"].dt.year == year].copy()
        # groupby time and quantile to calculate the equal-weighted return
        df_q = df.groupby(["time", "quantile"])["ret_w"].mean().reset_index()
        df = df.merge(mkt_w, on="time", how="left")
        df["ret_w"] = df["ret_w"] - df["cmkt_w"]
        df_q["cum_ret"] = (1 + df_q["ret_w"]).groupby(df_q["quantile"]).cumprod()
        cumret_plot_dict[ml][year] = df_q
