"""
Script to perform feature engineering on the data
"""

import re

import numpy as np
import pandas as pd

from environ.constants import PROCESSED_DATA_PATH

df = pd.read_csv(PROCESSED_DATA_PATH / "gecko_features.csv")
df["time"] = pd.to_datetime(df["time"])


# winsorize the data
def winsor(x: pd.Series, cut=[0.01, 0.99]) -> pd.Series:
    """
    Function to winsorize the data
    """
    x = x.copy()
    lb = x.quantile(cut[0])
    ub = x.quantile(cut[1])
    x[x < lb] = lb
    x[x > ub] = ub
    return x


# calculate the weekly return
df.sort_values(["id", "time"], ascending=True, inplace=True)
df["ret"] = df.groupby(["id"])["prices"].shift(-7) / df["prices"] - 1

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.drop(columns=["prc_i", "prc_j"], inplace=True)

info_vars = ["id", "year", "week", "time", "prices"]
ret_vars = ["ret"]
wret_vars = [f"{x}_w" for x in ret_vars]

factor_vars = [x for x in df.columns if re.findall("size_|vol_|volume_|mom_", x)]
wfactor_vars = [f"{x}_w" for x in factor_vars]

for var in [x for x in factor_vars + ret_vars]:
    df[f"{var}_w"] = winsor(df[var])

df = df.loc[:, info_vars + wret_vars + wfactor_vars]
df.dropna(inplace=True)

