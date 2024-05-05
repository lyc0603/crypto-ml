"""
Script to process the categorical dummy for Top 100 cryptocurrencies
"""

import warnings

import pandas as pd

from environ.constants import CATEGORY_LIST, PROCESSED_DATA_PATH

warnings.filterwarnings("ignore")

df_panel = pd.read_csv(PROCESSED_DATA_PATH / "panel_with_macro.csv")

df_token_list = pd.read_csv(PROCESSED_DATA_PATH / "token_lst.csv")
df_cate = pd.read_csv(PROCESSED_DATA_PATH / "category.csv", index_col=0)
df_cate = df_cate.T

for cate in CATEGORY_LIST:
    for token in df_cate[df_cate[cate] == 1].index.to_list():
        df_token_list.loc[df_token_list["id"] == token, cate] = 1

df_token_list = df_token_list.fillna(0)

# merge with panel
df_panel = pd.merge(df_panel, df_token_list, on="id", how="left")

# add cate_ for the column name
for cate in CATEGORY_LIST:
    df_panel.rename(columns={cate: f"cate_{cate}"}, inplace=True)

df_panel.to_csv(PROCESSED_DATA_PATH / "panel_with_macro_cate.csv", index=False)



