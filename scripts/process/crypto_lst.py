"""
Script to generate the list of cryptocurrencies with mcap
"""

import pandas as pd
from environ.utils.db import load_db
from environ.constants import PROCESSED_DATA_PATH, TEST_START_DATE

df = load_db(db_name="crypto_ml", collection_name="features")
df = df[["id", "time", "market_caps"]]
df["time"] = pd.to_datetime(df["time"])


panel = pd.read_csv(PROCESSED_DATA_PATH / "panel_with_macro_cate_google.csv")
panel["time"] = pd.to_datetime(panel["time"])

test_start_date = pd.to_datetime(TEST_START_DATE)
test_end_date = df["time"].max()
week_date = list(pd.date_range(test_start_date, test_end_date, freq="W"))
panel = panel.loc[panel["time"].isin(week_date)]

panel = panel.merge(df, on=["id", "time"], how="left")
panel = panel[["id", "time", "market_caps"]]
panel["rank"] = panel.groupby("time")["market_caps"].rank(ascending=False)

panel.to_csv(PROCESSED_DATA_PATH / "crypto_lst_with_mcap.csv", index=False)
