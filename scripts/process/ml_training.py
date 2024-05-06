"""
Script to train the machine learning model
"""


import sys

sys.path.append("/home/yichen/crypto-ml")

import os
import pickle
import re
import warnings

import pandas as pd
from tqdm import tqdm

from environ.constants import PARAM_GRID, PROCESSED_DATA_PATH, TEST_START_DATE
from environ.process.ml_tuning import enet, gbrt, lasso, nn, ols, rf, pls, pcr

warnings.filterwarnings("ignore")

# load the data
df = pd.read_csv(PROCESSED_DATA_PATH / "panel_with_macro_cate_google.csv")
df["time"] = pd.to_datetime(df["time"])
xvar = [x for x in df.columns if re.findall("size_|vol_|volume_|mom_|attn_", x)]
mvar = [x for x in df.columns if re.findall("macro_", x)]
cvar = [x for x in df.columns if re.findall("cate_", x)]

if not os.path.exists(PROCESSED_DATA_PATH / "res"):
    os.makedirs(PROCESSED_DATA_PATH / "res")

for path in ["model", "test", "params"]:
    if not os.path.exists(PROCESSED_DATA_PATH / "res" / path):
        os.makedirs(PROCESSED_DATA_PATH / "res" / path)

# split the date
test_start_date = pd.to_datetime(TEST_START_DATE)
test_end_date = df["time"].max()
week_date = list(pd.date_range(test_start_date, test_end_date, freq="W"))

# # OLS
# test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

# for date in tqdm(week_date):
#     date_str = date.strftime("%Y-%m-%d")
#     if os.path.exists(
#             PROCESSED_DATA_PATH / "res" / "test" / f"ols_category_{date_str}.csv"
#         ):
#             continue
#     # select the best model
#     test_dict, opt_model, params_list = ols(df, date, xvar, mvar, cvar)

#     # save models
#     with open(PROCESSED_DATA_PATH / "res" / "model" / f"ols_{date_str}.pkl", "wb") as f:
#         pickle.dump(opt_model, f)

#     # save the params
#     params_df = pd.DataFrame(params_list)
#     params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"ols_{date_str}.csv",
#         index=False,
#     )

#     # save the test
#     for var, test_df in test_dict.items():
#         test_df = pd.DataFrame(test_df)
#         test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"ols_{var}_{date_str}.csv",
#             index=False,
#         )

# PCR
test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

for date in tqdm(week_date):
    date_str = date.strftime("%Y-%m-%d")
    if os.path.exists(
            PROCESSED_DATA_PATH / "res" / "test" / f"pcr_category_{date_str}.csv"
        ):
            continue
    # select the best model
    test_dict, opt_model, params_list = pcr(df, date, xvar, mvar, cvar)

    # save models
    with open(PROCESSED_DATA_PATH / "res" / "model" / f"pcr_{date_str}.pkl", "wb") as f:
        pickle.dump(opt_model, f)

    # save the params
    params_df = pd.DataFrame(params_list)
    params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
    params_df.to_csv(
        PROCESSED_DATA_PATH / "res" / "params" / f"pcr_{date_str}.csv",
        index=False,
    )

    # save the test
    for var, test_df in test_dict.items():
        test_df = pd.DataFrame(test_df)
        test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
        test_df.to_csv(
            PROCESSED_DATA_PATH / "res" / "test" / f"pcr_{var}_{date_str}.csv",
            index=False,
        )

# # PLS
# test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

# for date in tqdm(week_date):
#     date_str = date.strftime("%Y-%m-%d")
#     if os.path.exists(
#             PROCESSED_DATA_PATH / "res" / "test" / f"pls_category_{date_str}.csv"
#         ):
#             continue
#     # select the best model
#     test_dict, opt_model, params_list = pls(df, date, xvar, mvar, cvar)

#     # save models
#     with open(PROCESSED_DATA_PATH / "res" / "model" / f"pls_{date_str}.pkl", "wb") as f:
#         pickle.dump(opt_model, f)

#     # save the params
#     params_df = pd.DataFrame(params_list)
#     params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"pls_{date_str}.csv",
#         index=False,
#     )

#     # save the test
#     for var, test_df in test_dict.items():
#         test_df = pd.DataFrame(test_df)
#         test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"pls_{var}_{date_str}.csv",
#             index=False,
#         )

# # Lasso
# test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

# for date in tqdm(week_date):
#     date_str = date.strftime("%Y-%m-%d")
#     if os.path.exists(
#             PROCESSED_DATA_PATH / "res" / "test" / f"lasso_category_{date_str}.csv"
#         ):
#             continue
#     # select the best model
#     test_dict, opt_model, params_list = lasso(df, date, xvar, mvar, cvar)

#     # save models
#     with open(PROCESSED_DATA_PATH / "res" / "model" / f"lasso_{date_str}.pkl", "wb") as f:
#         pickle.dump(opt_model, f)

#     # save the params
#     params_df = pd.DataFrame(params_list)
#     params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"lasso_{date_str}.csv",
#         index=False,
#     )

#     # save the test
#     for var, test_df in test_dict.items():
#         test_df = pd.DataFrame(test_df)
#         test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"lasso_{var}_{date_str}.csv",
#             index=False,
#         )

# # elastic net
# test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

# for date in tqdm(week_date):
#     date_str = date.strftime("%Y-%m-%d")
#     if os.path.exists(
#             PROCESSED_DATA_PATH / "res" / "test" / f"enet_category_{date_str}.csv"
#         ):
#             continue
#     print(date_str)
#     # select the best model
#     test_dict, opt_model, params_list = enet(df, date, xvar, mvar, cvar)

#     # save models
#     with open(PROCESSED_DATA_PATH / "res" / "model" / f"enet_{date_str}.pkl", "wb") as f:
#         pickle.dump(opt_model, f)

#     # save the params
#     params_df = pd.DataFrame(params_list)
#     params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"enet_{date_str}.csv",
#         index=False,
#     )

#     # save the test
#     for var, test_df in test_dict.items():
#         test_df = pd.DataFrame(test_df)
#         test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"enet_{var}_{date_str}.csv",
#             index=False,
#         )

# # Gradient boosting regressor
# test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

# for date in tqdm(week_date):
#     # select the best model
#     test_dict, opt_model, params_list = gbrt(df, date, xvar, mvar, cvar)
#     date_str = date.strftime("%Y-%m-%d")

#     # save models
#     with open(PROCESSED_DATA_PATH / "res" / "model" / f"gbrt_{date_str}.pkl", "wb") as f:
#         pickle.dump(opt_model, f)

#     # save the params
#     params_df = pd.DataFrame(params_list)
#     params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"gbrt_{date_str}.csv",
#         index=False,
#     )

#     # save the test
#     for var, test_df in test_dict.items():
#         test_df = pd.DataFrame(test_df)
#         test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"gbrt_{var}_{date_str}.csv",
#             index=False,
#         )

# # random forest
# test_dict_agg = {var: pd.DataFrame() for var in [""] + xvar + mvar + ["category"]}

# for date in tqdm(week_date):
#     # select the best model
#     test_dict, opt_model, params_list = rf(df, date, xvar, mvar, cvar)
#     date_str = date.strftime("%Y-%m-%d")

#     # save models
#     with open(PROCESSED_DATA_PATH / "res" / "model" / f"rf_{date_str}.pkl", "wb") as f:
#         pickle.dump(opt_model, f)

#     # save the params
#     params_df = pd.DataFrame(params_list)
#     params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#     params_df.to_csv(
#         PROCESSED_DATA_PATH / "res" / "params" / f"rf_{date_str}.csv",
#         index=False,
#     )

#     # save the test
#     for var, test_df in test_dict.items():
#         test_df = pd.DataFrame(test_df)
#         test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#         test_df.to_csv(
#             PROCESSED_DATA_PATH / "res" / "test" / f"rf_{var}_{date_str}.csv",
#             index=False,
#         )


# # neural network
# for hidden_layer_sizes in PARAM_GRID["nn"]["hidden_layer_sizes"]:
#     for date in tqdm(week_date, desc=f"hidden_layer_sizes: {len(hidden_layer_sizes)}"):
#         # check if the test already exists
#         date_str = date.strftime("%Y-%m-%d")
#         if os.path.exists(
#             PROCESSED_DATA_PATH / "res_comp" / "test" / f"nn_{len(hidden_layer_sizes)}_category_{date_str}.csv"
#         ):
#             continue
#         print(date_str)
#         # select the best model
#         test_dict, model_list, params_list = nn(
#             df, date, xvar, mvar, cvar, hidden_layer_sizes
#         )

#         # save models
#         for model_idx, opt_model in enumerate(model_list):
#             opt_model.save(
#                 PROCESSED_DATA_PATH
#                 / "res_comp"
#                 / "model"
#                 / f"nn_{len(hidden_layer_sizes)}_{date_str}_{model_idx}.h5"
#             )

#         params_df = pd.DataFrame(params_list)
#         params_df["date"] = params_df["date"].dt.strftime("%Y-%m-%d")
#         params_df.to_csv(
#             PROCESSED_DATA_PATH
#             / "res_comp"
#             / "params"
#             / f"nn_{len(hidden_layer_sizes)}_{date_str}.csv",
#             index=False,
#         )

#         for var, test_df in test_dict.items():
#             test_df = pd.DataFrame(test_df)
#             test_df["time"] = test_df["time"].dt.strftime("%Y-%m-%d")
#             test_df.to_csv(
#                 PROCESSED_DATA_PATH
#                 / "res_comp"
#                 / "test"
#                 / f"nn_{len(hidden_layer_sizes)}_{var}_{date_str}.csv",
#                 index=False,
#             )
