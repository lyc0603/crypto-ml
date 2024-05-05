"""
Script to process the google trend index
"""

import warnings

import pandas as pd
from tqdm import tqdm

from environ.constants import DATA_PATH, PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")

ggl_df = pd.read_csv(f"{DATA_PATH}/token_lst_google.csv")
ggl_lst = ggl_df.loc[ggl_df["status"].isna(), "id"].to_list()

df_panel = pd.read_csv(f"{PROCESSED_DATA_PATH}/panel_with_macro_cate.csv")
df_panel["time"] = pd.to_datetime(df_panel["time"])
df_panel["yyyyww"] = df_panel["time"].dt.strftime("%Y%U")

def load_attn(path: str) -> pd.DataFrame:
    """
    Function to load the google trend index for a given token
    """
    df = pd.read_csv(path, skiprows=1)
    df.columns = ["time", "google"]
    df["time"] = pd.to_datetime(df["time"])
    df.sort_values("time", ascending=True, inplace=True)
    df.replace("<1", 0, inplace=True)
    return df


df_ggl = pd.DataFrame()

for token in tqdm(ggl_lst):
    df_attn = load_attn(f"{DATA_PATH}/google/{token}.csv")
    start_point_time, start_point = df_attn.iloc[0]["time"], df_attn.iloc[0]["google"]
    try:
        df_attn_sup = load_attn(f"{DATA_PATH}/google_sup/{token}.csv")
        end_point_time, end_point = (
            df_attn_sup.iloc[-1]["time"],
            df_attn_sup.iloc[-1]["google"],
        )
        df_attn_sup = df_attn_sup.loc[df_attn_sup["time"] != start_point_time]
        if (start_point != 0) | (end_point != 0):
            ratio = start_point / end_point
            df_attn_sup["google"] = df_attn_sup["google"] * ratio
        df_attn = pd.concat([df_attn, df_attn_sup])
        df_attn["google"] = df_attn["google"] / df_attn["google"].max() * 100
        df_attn["google"] = df_attn["google"].apply(lambda x: round(x))
    except:
        pass

    df_attn["id"] = token
    df_attn.sort_values("time", ascending=True, inplace=True)
    df_attn["google_l1w"] = df_attn["google"].rolling(4).mean()
    df_attn["google_l1w"] = df_attn["google_l1w"].shift(1)
    df_attn.dropna(inplace=True)
    df_attn["google"] = df_attn["google"].apply(float) - df_attn["google_l1w"].apply(
        float
    )
    df_attn.drop(columns=["google_l1w"], inplace=True)
    df_ggl = pd.concat([df_ggl, df_attn])

df_ggl.sort_values("time", ascending=True, inplace=True)
# add six days to refect the end of the period
df_ggl["time"] = df_ggl["time"] + pd.DateOffset(days=6)
# add one week to avoid information leakage
df_ggl["time"] = df_ggl["time"] + pd.DateOffset(weeks=1)
df_ggl["yyyyww"] = df_ggl["time"].dt.strftime("%Y%U")
df_ggl = df_ggl[["yyyyww", "id", "google"]]

# merge the google trend index with the panel
df_panel = df_panel.merge(df_ggl, on=["yyyyww", "id"], how="left")
df_panel["google"] = df_panel["google"].fillna(0)
df_panel.drop(columns=["yyyyww"], inplace=True)
df_panel.rename(columns={"google": "attn_google"}, inplace=True)
df_panel.to_csv(f"{PROCESSED_DATA_PATH}/panel_with_macro_cate_google.csv", index=False)
