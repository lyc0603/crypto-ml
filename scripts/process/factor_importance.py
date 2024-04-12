"""
Script to generate data for factor importance
"""

import pandas as pd

from environ.constants import ML_METHOD, PROCESSED_DATA_PATH

factor_importance_dict = {}

for method in ML_METHOD:
    df_param = pd.read_csv(PROCESSED_DATA_PATH / "res" / "params" / f"{method}.csv")
    df_param.sort_values("date", ascending=True, inplace=True)
    df_param_importance = pd.DataFrame()

    for date in df_param["date"].unique():
        df_param_date = df_param.loc[df_param["date"] == date].copy()
        df_param_date["test_score"] = (
            df_param_date["test_score"]
            - df_param_date.loc[df_param_date["dvar"].isna(), "test_score"].values[0]
        )
        df_param_date = df_param_date.loc[~df_param_date["dvar"].isna()]
        df_param_date["test_score"] = (
            df_param_date["test_score"] - df_param_date["test_score"].min()
        ) / (df_param_date["test_score"].max() - df_param_date["test_score"].min())

        df_param_date["dvar"] = df_param_date["dvar"].str.replace("_w", "")

        df_param_date = df_param_date.pivot(
            index="date", columns="dvar", values="test_score"
        )
        df_param_importance = pd.concat([df_param_importance, df_param_date])

    df_param_importance = df_param_importance.T
    factor_importance_dict[method] = df_param_importance
    