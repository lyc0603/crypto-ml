"""
Script to calculate the importance of macro variables in predicting the target variable.
"""

import pandas as pd
import glob
from tqdm import tqdm

from environ.constants import ML_METHOD, PROCESSED_DATA_PATH, ML_PATH, ML_NAMING_DICT, MACRO_VARIABLES  

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
    df_param["dvar"].fillna("all", inplace=True)
    df_param = df_param.loc[df_param["dvar"].isin(["macro_" + _ for _ in MACRO_VARIABLES] + ["all"])]

    df_params = df_param.groupby("dvar")["train_score"].mean().reset_index()
    df_params["train_score"] = (
        df_params["train_score"]
        - df_params.loc[df_params["dvar"] == "all", "train_score"].values[0]
    )
    df_params = df_params.loc[~(df_params["dvar"] == "all")]
    total = df_params["train_score"].sum()
    df_params["train_score"] = df_params["train_score"] / total

    df_params["dvar"] = df_params["dvar"].str.replace("macro_", "")
    df_param_importance = df_params[["dvar", "train_score"]]
    df_param_importance["train_score"] = (
        df_param_importance["train_score"] * 100
    ).round(2).map("{:.2f}".format)
    factor_importance_dict[ML_NAMING_DICT[method]] = df_param_importance