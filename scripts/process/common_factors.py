"""
Script to perform feature engineering on the data
"""

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.linear_model import LinearRegression

pandarallel.initialize(progress_bar=True, nb_workers=36)

from environ.constants import PROCESSED_DATA_PATH

df = pd.read_csv(PROCESSED_DATA_PATH / "gecko_weekly.csv")
df["time"] = pd.to_datetime(df["time"])
df.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)

# size
df["size_mcap"] = df.groupby(["id"])["market_caps"].shift()
df["size_prc"] = df.groupby(["id"])["prices"].shift()
df["size_maxdprc"] = df.groupby(["id"])["max_prices"].shift()
df["size_age"] = df.groupby(["id"])["time"].rank().shift()

# momentum
for i, j in [
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (4, 1),
    (8, 0),
    (16, 0),
    (50, 0),
    (100, 0),
]:
    df["prc_i"] = df.groupby(["id"])["prices"].shift(
        i * 7 + 1
    )  # 注意要+1，否则就是用已有的信息去构建投资组合了，look-ahead bias
    df["prc_j"] = df.groupby(["id"])["prices"].shift(j * 7 + 1)
    df[f"mom_{i}_{j}"] = df["prc_j"] / df["prc_i"] - 1

# volume
df["volume_vol"] = df.groupby(["id"])["unit_volumes"].shift()
df["volume_vol"] = np.log(df["volume_vol"] + 1)  # 注意要+1，因为有些周交易量为0

df["volume_prcvol"] = df.groupby(["id"])["avg_volumes"].shift()
df["volume_prcvol"] = np.log(df["volume_prcvol"] + 1)

df["volume_volscaled"] = (
    df.groupby(["id"])["avg_volumes"].shift()
    / df.groupby(["id"])["market_caps"].shift()
)
df["volume_volscaled"] = np.log(df["volume_volscaled"] + 1)

# volatility
df["vol_retvol"] = df.groupby(["id"])["std_daily_ret"].shift()
df["vol_maxret"] = df.groupby(["id"])["max_daily_ret"].shift()
df["vol_stdprcvol"] = df.groupby(["id"])["std_volumes"].shift()
df["vol_damihud"] = (
    df.groupby(["id"])["avg_daily_ret"].shift().map(abs)
    / df.groupby(["id"])["avg_volumes"].shift()
)

# market feature
er = pd.read_csv(PROCESSED_DATA_PATH / "gecko_mkt.csv")
er["time"] = pd.to_datetime(er["time"])
df["time_l1"] = df.groupby(["id"])["time"].shift()
df["key"] = df[["id", "time_l1"]].values.tolist()


def cal_vol(key):
    """
    Function to calculate the market related features
    """
    idx, time = key
    time_l365 = time - pd.offsets.Day(365)
    tmp = er.loc[(er["id"] == idx) & er["time"].between(time_l365, time)].dropna()

    if tmp.shape[0] <= 60:
        return np.nan, np.nan, np.nan

    X1 = tmp[["cmkt"]].values
    X2 = tmp[["cmkt", "cmkt_l1", "cmkt_l2"]].values
    Y = tmp["eret"].values

    reg = LinearRegression().fit(X1, Y)
    beta = reg.coef_[0]
    idio = (Y - reg.predict(X1)).std()
    r1 = reg.score(X1, Y)

    reg = LinearRegression().fit(X2, Y)
    r2 = reg.score(X2, Y)
    delay = r2 - r1

    return beta, idio, delay


df["result"] = df["key"].parallel_apply(cal_vol)
df["vol_beta"], df["vol_idiovol"], df["vol_delay"] = zip(*list(df["result"].values))
df["vol_beta2"] = df["vol_beta"] ** 2
df.drop(columns=["time_l1", "key", "result"], inplace=True)
df.to_csv(PROCESSED_DATA_PATH / "gecko_features.csv", index=False)
