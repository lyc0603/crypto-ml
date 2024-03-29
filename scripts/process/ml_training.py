"""
Script to train the machine learning model
"""

import os
import warnings

import pandas as pd
from tqdm import tqdm

from environ.constants import PARAM_GRID, PROCESSED_DATA_PATH, TEST_START_DATE
from environ.process.ml_tuning import nn
from scripts.process.feature_engineering import df, wfactor_vars

warnings.filterwarnings("ignore")

if not os.path.exists(PROCESSED_DATA_PATH / "res"):
    os.makedirs(PROCESSED_DATA_PATH / "res")

for path in ["model", "test", "params"]:
    if not os.path.exists(PROCESSED_DATA_PATH / "res" / path):
        os.makedirs(PROCESSED_DATA_PATH / "res" / path)

# split the date
test_start_date = pd.to_datetime(TEST_START_DATE)
test_end_date = df["time"].max()
week_date = list(pd.date_range(test_start_date, test_end_date, freq="W"))

week_date = [test_start_date]

for hidden_layer_sizes in PARAM_GRID["nn"]["hidden_layer_sizes"]:
    params_df = pd.DataFrame()
    test_dict_agg = {var: pd.DataFrame() for var in [""] + wfactor_vars}

    for date in tqdm(week_date):
        # select the best model
        test_dict, model_list, params_list = nn(
            df, date, wfactor_vars, hidden_layer_sizes
        )
        date_str = date.strftime("%Y-%m-%d")

        # save models
        for model_idx, opt_model in enumerate(model_list):
            opt_model.save(
                PROCESSED_DATA_PATH
                / "res"
                / "model"
                / f"nn_{len(hidden_layer_sizes)}_{date_str}_{model_idx}.h5"
            )

        params_df = pd.concat([pd.DataFrame(params_list), params_df])
        for var, test_df in test_dict.items():
            test_dict_agg[var] = pd.concat([test_dict_agg[var], test_df])

    # save the params
    params_df.to_csv(
        PROCESSED_DATA_PATH / "res" / "params" / f"nn_{len(hidden_layer_sizes)}.csv",
        index=False,
    )
    for var, test_df in test_dict_agg.items():
        test_df.to_csv(
            PROCESSED_DATA_PATH / "res" / "test" / f"nn_{len(hidden_layer_sizes)}_{var}.csv",
            index=False,
        )
