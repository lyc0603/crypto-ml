"""
Function to clean the data
"""

import numpy as np
import pandas as pd
from tqdm import tqdm


def agg(
    df_crypto: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Function to aggregate the data
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)
    df_crypto[time_col] = pd.to_datetime(df_crypto[time_col])
    
    df_week = []

    # iterate through all data
    for crypto_id, crypto_df in tqdm(df_crypto.groupby("id")):
        # iterate through the data
        for _, row in crypto_df.loc[
            (crypto_df["time"] >= start_time) & (crypto_df["time"] < end_time)
        ].iterrows():

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
                "unit_volumes": (
                    row["total_volumes"] / row["prices"]
                    if row["prices"] != 0
                    else np.nan
                ),
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
                    "avg_daily_ret": weekly_data["daily_ret"].mean(),
                    "max_daily_ret": weekly_data["daily_ret"].max(),
                    "std_daily_ret": weekly_data["daily_ret"].std(),
                }
            for key, value in col_var_mapping.items():
                data_dict[key] = value
            # append the data
            df_week.append(data_dict)
    return pd.DataFrame(df_week)

