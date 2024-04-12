"""
Script to visualize the cumulative returns of a strategy
"""

import matplotlib.pyplot as plt

from environ.constants import FIGURE_PATH
from scripts.process.cumret import cumret_plot_dict

# plot the cumulative return
for ml, df_q in cumret_plot_dict.items():
    plt.figure(figsize=(6, 3))
    for quantile in range(1, 6, 1):
        df_q_q = df_q.loc[df_q["quantile"] == quantile]
        plt.plot(df_q_q["time"], df_q_q["cum_ret"], label=f"Quantile {quantile}")

    plt.legend(loc="upper right")
    plt.xticks(rotation=90)
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/cumret_{ml}.pdf")