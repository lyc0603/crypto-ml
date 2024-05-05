"""
Script to process weekly return heatmap
"""

import pandas as pd

from environ.constants import ML_METHOD
from scripts.process.portfolio import portfolio_dict
from scripts.process.capm import mkt

weekly_ret_dict = {}

for ml in ML_METHOD:
    df_quantile_ret = pd.DataFrame()
    for quantile in range(5, 0, -1):
        df = portfolio_dict[ml].loc[portfolio_dict[ml]["quantile"] == quantile].copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.groupby("time")["ret_w"].mean().reset_index()
        df = df.merge(mkt.reset_index()[["time", "cmkt"]], on="time", how="left")
        df["ret_w"] = df["ret_w"] - df["cmkt"]
        df["time"] = df["time"].dt.strftime("%Y-%m-%d")
        df.set_index("time", inplace=True)
        df_quantile_ret[quantile] = df["ret_w"]
    df_quantile_ret = df_quantile_ret.T

    weekly_ret_dict[ml] = df_quantile_ret



        