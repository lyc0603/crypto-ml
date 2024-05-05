"""
Script to evaluate the performance
"""

import glob

import matplotlib.pyplot as plt
import pandas as pd

from environ.constants import FIGURE_PATH, PROCESSED_DATA_PATH
# from scripts.process.sp_ret import sp_df

def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Function to calculate the r2 score
    """
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true) ** 2)

# test_rf = pd.read_csv(PROCESSED_DATA_PATH / "res_nolog" / "test" /"rf_.csv")
test_rf = pd.concat(
    [pd.read_csv(f) for f in glob.glob(str(PROCESSED_DATA_PATH / "res" / "test" / "pls__*.csv"))]
)

test_rf["time"] = pd.to_datetime(test_rf["time"])

# sample spliting
# test_rf = test_rf.loc[test_rf["time"] <= "2024-01-01"]
# test_rf = (
#     test_rf.groupby("time")
#     .apply(lambda x: x.nlargest(20, "size_mcap_w"))
#     .reset_index(drop=True)
# )

print(test_rf[["log_eret_w", "log_eret_pred"]].describe())

# calculate the r2 score of the model
score = r2_score(test_rf["log_eret_w"], test_rf["log_eret_pred"])
print(score)

# calculate the score of the model with time
plt.plot(test_rf.groupby("time").apply(
    lambda x: r2_score(x["log_eret_w"], x["log_eret_pred"])
))
plt.xticks(rotation=45)
plt.show()
print(
    test_rf.groupby("time").apply(
    lambda x: r2_score(x["log_eret_w"], x["log_eret_pred"])
).mean()
)

# calculate the quantile of ret_pred for each time
for time in test_rf["time"].unique():
    test_rf.loc[test_rf["time"] == time, "quantile"] = pd.qcut(
        test_rf.loc[test_rf["time"] == time, "log_eret_pred"], 5, labels=False, duplicates="drop"
    )

df_q = pd.DataFrame()

for quantile in range(5):
    q = (
        test_rf.loc[test_rf["quantile"] == quantile]
        .groupby(["time"])["ret_w"]
        .mean()
        .reset_index()
    )
    q["cum_ret"] = (1 + q["ret_w"]).cumprod()
    q["avg_ret"] = q["ret_w"].mean()
    # df_q[quantile] = q["ret_w"].values
    print(quantile, q["avg_ret"].mean())
    # plt.plot(q["time"], q["ret_w"], label=quantile)
    plt.plot(q["time"], q["cum_ret"], label=quantile)

# # calculate how many times the quantile 4 is the best
# df_q["time"] = q["time"].values
# df_q.set_index("time", inplace=True)
# df_q["best"] = df_q.idxmax(axis=1)
# print(df_q["best"].value_counts())

# # plot the bitcoin return
# btc = test_rf.loc[test_rf["id"] == "bitcoin"]
# btc["cum_ret"] = (1 + btc["ret_w"]).cumprod()
# plt.plot(btc["time"], btc["cum_ret"], label="bitcoin")

# # plot the sp return
# sp_df = sp_df.loc[(sp_df["date"] >= test_rf["time"].min()) & (sp_df["date"] <= test_rf["time"].max())]
# sp_df["cum_ret"] = (1 + sp_df["ret"]).cumprod()
# plt.plot(sp_df["date"], sp_df["cum_ret"], label="S&P")

plt.legend(title="Quantile")
plt.xticks(rotation=45)
plt.title("Performance of equal-weiged machine learning portfolios")
plt.ylabel("Cumulative return")
plt.show()
plt.savefig(FIGURE_PATH / "performance.png")

# calculate the quantile of ret_pred
test_rf["quantile"] = test_rf.groupby(["time"])["log_eret_pred"].transform(
    lambda x: pd.qcut(x, 5, labels=False, duplicates="drop")
)
for quantile in range(5):
    q = test_rf.loc[test_rf["quantile"] == quantile]
    q["ret_m"] = q["ret_w"] * q["size_mcap_w"]
    q = q.groupby(["time"])[["ret_m", "size_mcap_w"]].sum().reset_index()
    q["ret_m"] = q["ret_m"] / q["size_mcap_w"]
    q["cum_ret"] = (1 + q["ret_m"]).cumprod()
    plt.plot(q["time"], q["cum_ret"], label=quantile)
plt.legend(title="Quantile")
plt.xticks(rotation=45)
plt.title("Performance of mcap-weiged machine learning portfolios")
plt.ylabel("Cumulative return")
plt.show()

# # calculate the equal-weighed cumulative return
# test_rf = test_rf.loc[test_rf["ret_pred"] > 0]

# ewr = test_rf.groupby(["time"])["ret_w"].mean().reset_index()
# ewr["cum_ret"] = (1 + ewr["ret_w"]).cumprod()
# plt.plot(ewr["time"], ewr["cum_ret"], label="equal-weighed")


# # calculate the marketcap-weighed cumulative return
# mwr = test_rf.copy()
# mwr["ret_m"] = mwr["ret_w"] * mwr["size_mcap_w"]
# mwr = mwr.groupby(["time"])["ret_m", "size_mcap_w"].sum().reset_index()
# mwr["ret_m"] = mwr["ret_m"] / mwr["size_mcap_w"]
# mwr["cum_ret"] = (1 + mwr["ret_m"]).cumprod()
# plt.plot(mwr["time"], mwr["cum_ret"], label="mcap-weighted")
# plt.xticks(rotation=45)
# plt.title("Performance of positive prediction return machine learning portfolio")
# plt.ylabel("Cumulative return")
# plt.legend(title="weighting scheme")
# plt.show()
