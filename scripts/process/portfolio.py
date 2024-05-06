"""
Script to generate decile portfolio
"""

import numpy as np
import glob
import warnings
import pandas as pd
from tqdm import tqdm
from environ.constants import PROCESSED_DATA_PATH, ML_METHOD, ML_PATH

warnings.filterwarnings("ignore")

portfolio_dict = {}

for ml in tqdm(ML_METHOD):
    df = pd.concat(
        [
            pd.read_csv(file)
            for file in glob.glob(
                f"{PROCESSED_DATA_PATH}/{ML_PATH[ml]}/test/{ml}__*.csv"
            )
        ]
    )
    df.sort_values(["time", "log_eret_pred"], ascending=True, inplace=True)
    # convert the log_eret_pred to eret_pred
    df["eret_pred"] = df["log_eret_pred"].apply(lambda x: np.exp(x) - 1)
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
            df_q = pd.concat([df_q, df_quantile])

    portfolio_dict[ml] = df_q
