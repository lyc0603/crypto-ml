"""
Script to evaluate the performance
"""

import matplotlib.pyplot as plt
import pandas as pd

from environ.constants import PROCESSED_DATA_PATH
from environ.process.ml_model import r2_score

test_rf = pd.read_csv(PROCESSED_DATA_PATH / "test_nn.csv")
test_rf["time"] = pd.to_datetime(test_rf["time"])

# calculate the r2 score of the model
score = r2_score(test_rf["ret_w"], test_rf["ret_pred"])
print(score)

# calculate the r2 score with time
# test_rf.groupby(["time"])["ret_w", "ret_pred"].apply(
#     lambda x: r2_score(x["ret_w"], x["ret_pred"])
# ).plot()
# plt.show()

# calculate the quantile of ret_pred
test_rf["quantile"] = pd.qcut(test_rf["ret_pred"], 5, labels=False)
for quantile in range(5):
    q = (
        test_rf.loc[test_rf["quantile"] == quantile]
        .groupby(["time"])["ret_w"]
        .mean()
        .reset_index()
    )
    q["cum_ret"] = (1 + q["ret_w"]).cumprod()
    plt.plot(q["time"], q["cum_ret"], label=quantile)
plt.legend(title="Quantile")
plt.xticks(rotation=45)
plt.title("Performance of equal-weiged machine learning portfolios")
plt.ylabel("Cumulative return")
plt.show()

# calculate the quantile of ret_pred
test_rf["quantile"] = pd.qcut(test_rf["ret_pred"], 5, labels=False)
for quantile in range(5):
    q = test_rf.loc[test_rf["quantile"] == quantile]
    q["ret_m"] = q["ret_w"] * q["size_mcap_w"]
    q = q.groupby(["time"])["ret_m", "size_mcap_w"].sum().reset_index()
    q["ret_m"] = q["ret_m"] / q["size_mcap_w"]
    q["cum_ret"] = (1 + q["ret_m"]).cumprod()
    plt.plot(q["time"], q["cum_ret"], label=quantile)
plt.legend(title="Quantile")
plt.xticks(rotation=45)
plt.title("Performance of mcap-weiged machine learning portfolios")
plt.ylabel("Cumulative return")
plt.show()

# calculate the equal-weighed cumulative return
test_rf = test_rf.loc[test_rf["ret_pred"] > 0]

ewr = test_rf.groupby(["time"])["ret_w"].mean().reset_index()
ewr["cum_ret"] = (1 + ewr["ret_w"]).cumprod()
plt.plot(ewr["time"], ewr["cum_ret"], label="equal-weighed")


# calculate the marketcap-weighed cumulative return
mwr = test_rf.copy()
mwr["ret_m"] = mwr["ret_w"] * mwr["size_mcap_w"]
mwr = mwr.groupby(["time"])["ret_m", "size_mcap_w"].sum().reset_index()
mwr["ret_m"] = mwr["ret_m"] / mwr["size_mcap_w"]
mwr["cum_ret"] = (1 + mwr["ret_m"]).cumprod()
plt.plot(mwr["time"], mwr["cum_ret"], label="mcap-weighted")
plt.xticks(rotation=45)
plt.title("Performance of positive prediction return machine learning portfolio")
plt.ylabel("Cumulative return")
plt.legend(title="weighting scheme")
plt.show()
