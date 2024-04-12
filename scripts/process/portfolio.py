"""
Script to generate decile portfolio
"""

import warnings
import pandas as pd
from environ.constants import PROCESSED_DATA_PATH, ML_METHOD

warnings.filterwarnings("ignore")

portfolio_dict = {}

for ml in ML_METHOD:
    df = pd.read_csv(PROCESSED_DATA_PATH / "res" / "test" / f"{ml}_.csv")
    df.sort_values(["time", "ret_pred"], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df_q = pd.DataFrame()
    for time in df["time"].unique():
        df_time = df.loc[df["time"] == time].copy().reset_index(drop=True)
        for quantile in range(1, 6, 1):
            df_quantile = df_time.iloc[
                int(len(df_time) * (quantile - 1) / 5) : int(
                    len(df_time) * quantile / 5
                )
            ]

            df_quantile["quantile"] = quantile
            df_quantile["avg_ret"] = df_quantile["ret_w"].mean()

            df_q = pd.concat([df_q, df_quantile])

    portfolio_dict[ml] = df_q
