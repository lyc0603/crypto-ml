"""
Script to generate data for factor importance
"""

import pandas as pd
import glob
from tqdm import tqdm

from environ.constants import ML_METHOD, PROCESSED_DATA_PATH, ML_PATH

factor_importance_dict = {}

for method in tqdm(ML_METHOD):
    df_param = pd.concat(
        [
            pd.read_csv(f)
            for f in glob.glob(
                str(PROCESSED_DATA_PATH / ML_PATH[method] / "params" / f"{method}_*.csv")
            )
        ]
    )
    df_param.sort_values("date", ascending=True, inplace=True)
    df_param_importance = pd.DataFrame()

    for date in df_param["date"].unique():
        df_param_date = df_param.loc[df_param["date"] == date].copy()
        df_param_date["train_score"] = (
            df_param_date["train_score"]
            - df_param_date.loc[df_param_date["dvar"].isna(), "train_score"].values[0]
        )
        df_param_date = df_param_date.loc[~df_param_date["dvar"].isna()]
        df_param_date["train_score"] = (
            df_param_date["train_score"] - df_param_date["train_score"].min()
        ) / (df_param_date["train_score"].max() - df_param_date["train_score"].min())

        df_param_date["dvar"] = df_param_date["dvar"].str.replace("_w", "")

        df_param_date = df_param_date.pivot(
            index="date", columns="dvar", values="train_score"
        )
        df_param_importance = pd.concat([df_param_importance, df_param_date])

    df_param_importance = df_param_importance.T
    factor_importance_dict[method] = df_param_importance
    