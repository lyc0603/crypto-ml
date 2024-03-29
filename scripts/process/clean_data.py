"""
Script to process the crypto data
"""

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from environ.constants import PROCESSED_DATA_PATH
from scripts.fetch.stable import stablecoins

pandarallel.initialize(progress_bar=False, nb_workers=36)

# load the data after merging raw data
df_crypto = pd.read_csv(PROCESSED_DATA_PATH / "gecko_raw_merged.csv")

# remove the 0 timestamp
df_crypto = df_crypto[df_crypto["timestamp"] != 0]

# convert the timestamp to datetime and remove the current timestamp
df_crypto.sort_values(["id", "timestamp"], ascending=True, inplace=True)
df_crypto["date"] = pd.to_datetime(df_crypto["timestamp"], unit="ms")
df_crypto["date"] = df_crypto["date"].dt.strftime("%Y-%m-%d")
df_crypto.drop_duplicates(subset=["id", "date"], inplace=True, keep="first")
df_crypto.drop(columns=["timestamp"], inplace=True)
df_crypto["date"] = pd.to_datetime(df_crypto["date"])

# minus the date by 1
df_crypto["date"] = df_crypto["date"] - pd.Timedelta(days=1)
df_crypto = df_crypto.loc[df_crypto["date"] != df_crypto["date"].max()]

df_crypto.rename(
    columns={
        "date": "time",
        "price": "prices",
        "mcap": "market_caps",
        "vol": "total_volumes",
    },
    inplace=True,
)

# remove the stablecoins
df_crypto = df_crypto[~df_crypto["id"].isin(stablecoins)]

# only keep the top 100 market cap crypto every day using parrallel apply
df_crypto = (
    df_crypto.groupby("time")
    .parallel_apply(lambda x: x.nlargest(100, "market_caps"))
    .reset_index(drop=True)
)
df_crypto.sort_values(["id", "time"], ascending=True, inplace=True)


# calculate the daily return
df_crypto["daily_ret"] = df_crypto.groupby(["id"])["prices"].pct_change()
pl1 = df_crypto.groupby(["id"])["prices"].shift()
df_crypto.loc[pl1 == 0, "daily_ret"] = np.nan

# nan
df_crypto.dropna(
    subset=["prices", "market_caps", "total_volumes"], how="any", inplace=True
)


# convert the daily data to weekly data
df_crypto[["year", "week", "day"]] = df_crypto["time"].dt.isocalendar()
df_crypto.sort_values(["id", "year", "week", "day"], ascending=True, inplace=True)

# save the daily data
df_crypto.to_csv(PROCESSED_DATA_PATH / "gecko_daily.csv", index=False)

df_week = []

# iterate through all data
for crypto_id, crypto_df in tqdm(df_crypto.groupby("id")):
    # iterate through the data
    for row_idx, row in crypto_df.iterrows():

        data_dict = {}

        # get the week
        time, year, week, day = row["time"], row["year"], row["week"], row["day"]

        # get the weekly data
        weekly_data = crypto_df[
            (crypto_df["time"] >= time - pd.Timedelta(days=6))
            & (crypto_df["time"] <= time)
        ]

        col_var_mapping_common = {
            "time": time,
            "id": crypto_id,
            "year": year,
            "week": week,
            "day": day,
            "prices": row["prices"],
            "market_caps": row["market_caps"],
            "eow_volumes": row["total_volumes"],
            "unit_volumes" : row["total_volumes"] / row["prices"] if row["prices"] != 0 else np.nan,
        }

        if len(weekly_data) != 7:
            col_var_mapping = {
                **col_var_mapping_common,
                "max_prices": np.nan,
                "avg_volumes": np.nan,
                "std_volumes": np.nan,
                "avg_daily_ret": np.nan,
                "max_daily_ret": np.nan,
                "std_daily_ret": np.nan,
            }

        else:
            # get the weekly data
            col_var_mapping = {
                **col_var_mapping_common,
                "max_prices": weekly_data["prices"].max(),
                "avg_volumes": weekly_data["total_volumes"].mean(),
                "std_volumes": weekly_data["total_volumes"].std(),
                "avg_daily_ret" : weekly_data["daily_ret"].mean(),
                "max_daily_ret" : weekly_data["daily_ret"].max(),
                "std_daily_ret" : weekly_data["daily_ret"].std(),
            }
            
        for key, value in col_var_mapping.items():
            data_dict[key] = value
        # append the data
        df_week.append(data_dict)

# convert the data to dataframe
df_week = pd.DataFrame(df_week)
df_week.to_csv(PROCESSED_DATA_PATH / "gecko_weekly.csv")
