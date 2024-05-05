"""
Script to visualize the weekly returns of the stocks in the portfolio
"""

import matplotlib.pyplot as plt
import seaborn as sns

from environ.constants import FIGURE_PATH, ML_METHOD
from scripts.process.weekly_ret import weekly_ret_dict

plt.rcParams["figure.figsize"] = (8, 5)

for method in ML_METHOD:
    df = weekly_ret_dict[method]
    print(method)
    # print the mean of each quantile
    print(df.mean(axis=1))
    norm = plt.Normalize(-abs(df.values).max(), abs(df.values).max())
    sns.heatmap(df, cmap="seismic", norm=norm, cbar=True)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.xticks(rotation=90)
    plt.show()
    # plt.savefig(f"{FIGURE_PATH}/fi_{method}.pdf")