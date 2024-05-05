"""
Script to visualize the cumulative returns of a strategy
"""

import matplotlib.pyplot as plt

from environ.constants import FIGURE_PATH
from scripts.process.cumret_full import cumret_plot_dict, ls_plot_dict

# # plot the cumulative return
# for ml, ml_plot in cumret_plot_dict.items():
#     print(ml)
#     plt.figure(figsize=(3, 3))
#     for quantile in range(1, 6, 1):
#         quantile_plot = ml_plot.loc[ml_plot["quantile"] == quantile].copy()
#         plt.plot(quantile_plot["time"], quantile_plot["cum_ret"], label=f"Quantile {quantile}")

#     plt.legend(loc="upper right")
#     plt.xticks(rotation=90)
#     plt.grid(alpha=0.5)
#     plt.tight_layout()
#     plt.show()
#     # plt.savefig(f"{FIGURE_PATH}/cumret_{ml}.pdf")

# plot the long-short strategy
for ml, ml_plot in ls_plot_dict.items():
    print(ml)
    plt.figure(figsize=(3, 3))
    plt.plot(ml_plot["time"], ml_plot["cum_ret"], label="Long-Short")
    plt.legend(loc="upper right")
    plt.xticks(rotation=90)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()
    # plt.savefig(f"{FIGURE_PATH}/ls_{ml}.pdf")