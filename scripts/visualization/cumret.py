"""
Script to visualize the cumulative returns of a strategy
"""

import matplotlib.pyplot as plt

from environ.constants import FIGURE_PATH
from scripts.process.cumret import cumret_plot_dict

# plot the cumulative return
for ml, ml_plot_dict in cumret_plot_dict.items():
    print(ml)
    for year, year_plot in ml_plot_dict.items():
        plt.figure(figsize=(3, 3))
        for quantile in range(1, 6, 1):
            quantile_plot = year_plot.loc[year_plot["quantile"] == quantile].copy()
            plt.plot(quantile_plot["time"], quantile_plot["cum_ret"], label=f"Quantile {quantile}")

        plt.legend(loc="upper right")
        plt.xticks(rotation=90)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{FIGURE_PATH}/cumret_{ml}.pdf")