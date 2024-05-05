"""
Script to calculate the R2 score of the model
"""

import glob

import pandas as pd
from tqdm import tqdm

from environ.constants import (ML_METHOD, ML_NAMING_DICT, ML_PATH,
                               PROCESSED_DATA_PATH)

crypto_lst = pd.read_csv(PROCESSED_DATA_PATH / "crypto_lst_with_mcap.csv")

def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Function to calculate the r2 score
    """
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true) ** 2)

r2_dict = {}

for method in tqdm(ML_METHOD):
    method_name = ML_NAMING_DICT[method]
    r2_dict[method_name] = {}
    test_df = pd.concat(
        [
            pd.read_csv(f)
            for f in glob.glob(
                str(PROCESSED_DATA_PATH / ML_PATH[method] / "test" / f"{method}__*.csv")
            )
        ]
    )
    test_df = test_df.merge(crypto_lst, on=["id", "time"], how="left")
    r2_dict[method_name]["All"] = r2_score(test_df["log_eret_w"], test_df["log_eret_pred"])
    r2_dict[method_name]["Top"] = r2_score(
        test_df[test_df["rank"] <= 10]["log_eret_w"],
        test_df[test_df["rank"] <= 10]["log_eret_pred"],
    )
    r2_dict[method_name]["Bottom"] = r2_score(
        test_df[test_df["rank"] > 90]["log_eret_w"],
        test_df[test_df["rank"] > 90]["log_eret_pred"],
    )


