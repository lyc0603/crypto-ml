"""
Script to calculate the cumulative return
"""
import matplotlib.pyplot as plt
import pandas as pd

from environ.constants import ML_METHOD
from scripts.process.portfolio import portfolio_dict

cumret_plot_dict = {}

for ml in ML_METHOD:
    df = portfolio_dict[ml]
    df["time"] = pd.to_datetime(df["time"])
    # groupby time and quantile to calculate the equal-weighted return
    df_q = df.groupby(["time", "quantile"])["ret_w"].mean().reset_index()
    # calculate the cumulative return
    df_q["cum_ret"] = (1 + df_q["ret_w"]).groupby(df_q["quantile"]).cumprod() - 1
    cumret_plot_dict[ml] = df_q
