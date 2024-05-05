"""
Script to process the macro data
"""

import numpy as np
import pandas as pd

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

macro = pd.read_csv(DATA_PATH / "macro" / "PredictorData2023.xlsx - Monthly.csv")
df_panel = pd.read_csv(PROCESSED_DATA_PATH / "gecko_panel.csv")

df_panel["time"] = pd.to_datetime(df_panel["time"])
df_panel["yyyymm"] = df_panel["time"] - pd.DateOffset(months=1)
df_panel["yyyymm"] = df_panel["yyyymm"].dt.strftime("%Y%m").astype(int)


# convert Index to string and remove the , to .
macro["Index"] = macro["Index"].astype(str)
macro["Index"] = macro["Index"].str.replace(",", "")
macro["Index"] = macro["Index"].astype(float)

# construct macro data
macro["dp"] = np.log(macro["D12"]) - np.log(macro["Index"])
macro["ep"] = np.log(macro["E12"]) - np.log(macro["Index"])
macro["tms"] = macro["lty"] - macro["tbl"]
macro["dfy"] = macro["BAA"] - macro["AAA"]
macro.rename(
    columns={
        "dp": "macro_dp",
        "ep": "macro_ep",
        "b/m": "macro_bm",
        "ntis": "macro_ntis",
        "tbl": "macro_tbl",
        "tms": "macro_tms",
        "dfy": "macro_dfy",
        "svar": "macro_svar",
    },
    inplace=True,
)

macro = macro[
    [
        "yyyymm",
        "macro_dp",
        "macro_ep",
        "macro_bm",
        "macro_ntis",
        "macro_tbl",
        "macro_tms",
        "macro_dfy",
        "macro_svar",
    ]
]

# merge with panel
df_panel = pd.merge(df_panel, macro, on=["yyyymm"], how="left")
df_panel.drop(columns=["yyyymm"], inplace=True)
df_panel.dropna(inplace=True)
df_panel.to_csv(PROCESSED_DATA_PATH / "panel_with_macro.csv", index=False)
