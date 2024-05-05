"""
Script to calculate the importance of macro variables in predicting the target variable.
"""

import pandas as pd
import glob
from tqdm import tqdm

from environ.constants import (
    ML_METHOD,
    PROCESSED_DATA_PATH,
    ML_PATH,
    ML_NAMING_DICT,
    MACRO_VARIABLES,
    VAR_NAMING_DICT,
    H_ML
)

factor_importance_df = pd.DataFrame()

for method in tqdm(ML_METHOD):
    df_param = pd.concat(
        [
            pd.read_csv(f)
            for f in glob.glob(
                str(
                    PROCESSED_DATA_PATH / ML_PATH[method] / "params" / f"{method}_*.csv"
                )
            )
        ]
    )
    df_param["dvar"].fillna("all", inplace=True)
    df_param = df_param.loc[
        ~(df_param["dvar"].isin(["macro_" + _ for _ in MACRO_VARIABLES]))
    ]

    for remove in ["_w", "size_", "vol_", "volume_", "attn_"]:
        df_param["dvar"] = df_param["dvar"].str.replace(remove, "")

    df_param["dvar"] = df_param["dvar"].map(VAR_NAMING_DICT)

    df_params = df_param.groupby("dvar")["train_score"].mean().reset_index()
    df_params["train_score"] = (
        df_params["train_score"]
        - df_params.loc[df_params["dvar"] == "all", "train_score"].values[0]
    )
    df_params = df_params.loc[~(df_params["dvar"] == "all")]
    total = df_params["train_score"].sum()
    df_params["train_score"] = df_params["train_score"] / total

    df_params["dvar"] = df_params["dvar"].str.replace("macro_", "")
    df_params = df_params[["dvar", "train_score"]]
    df_params["method"] = ML_NAMING_DICT[method]
    factor_importance_df = pd.concat([factor_importance_df, df_params])

factor_importance_df = factor_importance_df.pivot(
    index="dvar", columns="method", values="train_score"
)

# calculate the rank of the macro variables across the methods
factor_importance_df["rank"] = factor_importance_df.rank(axis=0).mean(axis=1)
factor_importance_df.sort_values("rank", ascending=False, inplace=True)
factor_importance_df.drop(columns="rank", inplace=True)
factor_importance_df = factor_importance_df[[ML_NAMING_DICT[_] for _ in ML_METHOD]]
for key in factor_importance_df.columns:
    if key in H_ML:
        factor_importance_df.rename(columns={key: key + "+H"}, inplace=True)
