"""
Script to visualize the performance of the portfolio
"""

import pandas as pd

from environ.constants import ML_METHOD
from scripts.process.portfolio import portfolio_dict

asset_pricing_dict = {}

for ml in ML_METHOD:
    asset_pricing_dict[ml] = {}
    df = portfolio_dict[ml]
    df["time"] = pd.to_datetime(df["time"])

    for quantile in range(1, 6, 1):
        df_q = df.loc[df["quantile"] == quantile].copy()
        # percentage and two decimal points
        asset_pricing_dict[ml][quantile] = {
            "Avg": round(df_q["ret_w"].mean() * 100, 2),
            "Pred": round(df_q["ret_pred"].mean() * 100, 2),
            "SD": round(df_q["ret_w"].std() * 100, 2),
            "SR": round(df_q["ret_w"].mean() / df_q["ret_w"].std(), 2)
        }
