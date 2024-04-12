"""
Script to train the machine learning model
"""

import os
import re
import warnings
import pickle

import pandas as pd
from tqdm import tqdm

from environ.constants import PARAM_GRID, PROCESSED_DATA_PATH, TEST_START_DATE
from environ.process.ml_tuning import nn, rf

warnings.filterwarnings("ignore")


# load the data
df = pd.read_csv(PROCESSED_DATA_PATH / "gecko_panel.csv")
df["time"] = pd.to_datetime(df["time"])
wfactor_vars = [x for x in df.columns if re.findall("size_|vol_|volume_|mom_", x)]

if not os.path.exists(PROCESSED_DATA_PATH / "res"):
    os.makedirs(PROCESSED_DATA_PATH / "res")

for path in ["model", "test", "params"]:
    if not os.path.exists(PROCESSED_DATA_PATH / "res" / path):
        os.makedirs(PROCESSED_DATA_PATH / "res" / path)

# split the date
test_start_date = pd.to_datetime(TEST_START_DATE)
test_end_date = df["time"].max()
week_date = list(pd.date_range(test_start_date, test_end_date, freq="W"))

# # random forest
# params_df = pd.DataFrame()
# test_dict_agg = {var: pd.DataFrame() for var in [""] + wfactor_vars}

# for date in tqdm(week_date):
#     # select the best model
#     test_dict, opt_model, params_list = rf(
#         df, date, wfactor_vars,
#     )
#     date_str = date.strftime("%Y-%m-%d")

#     # save models
#     with open(
#         PROCESSED_DATA_PATH / "res" / "model" / f"rf_{date_str}.pkl", "wb"
#     ) as f:
#         pickle.dump(opt_model, f)

#     params_df = pd.concat([pd.DataFrame(params_list), params_df])
#     for var, test_df in test_dict.items():
#         test_dict_agg[var] = pd.concat([test_dict_agg[var], test_df])

# # save the params
# params_df.to_csv(
#     PROCESSED_DATA_PATH
#     / "res"
#     / "params"
#     / f"rf.csv",
#     index=False,
# )
# for var, test_df in test_dict_agg.items():
#     test_df.to_csv(
#         PROCESSED_DATA_PATH
#         / "res"
#         / "test"
#         / f"rf_{var}.csv",
#         index=False,
#     )

# # neural network
# for hidden_layer_sizes in PARAM_GRID["nn"]["hidden_layer_sizes"]:
#     params_df = pd.DataFrame()
#     test_dict_agg = {var: pd.DataFrame() for var in [""] + wfactor_vars}

#     for date in tqdm(week_date):
#         # select the best model
#         test_dict, model_list, params_list = nn(
#             df, date, wfactor_vars, hidden_layer_sizes
#         )
#         date_str = date.strftime("%Y-%m-%d")

#         # save models
#         for model_idx, opt_model in enumerate(model_list):
#             opt_model.save(
#                 PROCESSED_DATA_PATH
#                 / "res"
#                 / "model"
#                 / f"nn_{len(hidden_layer_sizes)}_{date_str}_{model_idx}.h5"
#             )

#         params_df = pd.concat([pd.DataFrame(params_list), params_df])
#         for var, test_df in test_dict.items():
#             test_dict_agg[var] = pd.concat([test_dict_agg[var], test_df])

#     # save the params
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"nn_{len(hidden_layer_sizes)}.csv",
#         index=False,
#     )
#     for var, test_df in test_dict_agg.items():
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"nn_{len(hidden_layer_sizes)}_{var}.csv",
#             index=False,
#         )
