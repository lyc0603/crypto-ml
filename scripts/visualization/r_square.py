"""
Script to calculate the R^2 value of the model
"""

import matplotlib.pyplot as plt
import pandas as pd

from environ.constants import ML_METHOD, PROCESSED_DATA_PATH, FIGURE_PATH
from environ.process.ml_model import r2_score
from scripts.process.portfolio import portfolio_dict

for ml in ML_METHOD:
    # figure size 
    plt.figure(figsize=(6, 3))
    df = portfolio_dict[ml]
    df["time"] = pd.to_datetime(df["time"])

    df.groupby("time").apply(
        lambda x: r2_score(x["ret_w"], x["ret_pred"])
    )

    plt.plot(df.groupby("time").apply(
        lambda x: r2_score(x["ret_w"], x["ret_pred"])
    ), label="All", alpha=0.75)
    plt.plot(df.loc[df["quantile"] == df["quantile"].max()].groupby("time").apply(
        lambda x: r2_score(x["ret_w"], x["ret_pred"])
    ), label="Top", alpha=0.75)
    plt.plot(df.loc[df["quantile"] == df["quantile"].min()].groupby("time").apply(
        lambda x: r2_score(x["ret_w"], x["ret_pred"])
    ), label="Bottom", alpha=0.75)

    plt.legend(loc="upper right")
    plt.xticks(rotation=90)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/r2_{ml}.pdf")