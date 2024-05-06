"""
Script to plot the portfolio return
"""

from scripts.process.cumret import cumret_plot_dict
import matplotlib.pyplot as plt
from environ.constants import FIGURE_PATH, ML_METHOD, ML_NAMING_DICT

PLOT_DICT = {
    "ols": {
        "color": "green",
        "ls": "dashed",
    },
    "lasso": {
        "color": "purple",
        "ls": "solid",
    },
    "enet": {
        "color": "red",
        "ls": "solid",
    },
    "rf": {
        "color": "blue",
        "ls": "solid",
    },
    "nn_1": {
        "color": "darkblue",
        "ls": "dashed",
    },
    "nn_2": {
        "color": "orange",
        "ls": "dashed",
    },
    "nn_3": {
        "color": "olive",
        "ls": "dashed",
    },
    "nn_4": {
        "color": "pink",
        "ls": "dashed",
    },
    "nn_5": {
        "color": "brown",
        "ls": "dashed",
    },
}

# plot the cumulative return
for port in ["l", "ls"]:
    plt.figure(figsize=(10, 6))
    for ml in ML_METHOD:
        ml_plot = cumret_plot_dict[ml]
        plt.plot(
            ml_plot.index,
            ml_plot[port],
            label=ML_NAMING_DICT[ml],
            color=PLOT_DICT[ml]["color"],
            linestyle=PLOT_DICT[ml]["ls"],
            alpha=0.75,
        )

    # plot the equal weight
    ml_plot = cumret_plot_dict["eq"]
    plt.plot(
        ml_plot["eq"],
        label="Equal",
        color="black",
        linestyle="dashed",
        alpha=0.75,
    )

    # plot the mcap weight
    ml_plot = cumret_plot_dict["mcap"]
    plt.plot(
        ml_plot["mcap"],
        label="Mcap",
        color="black",
        linestyle="dotted",
        alpha=0.75,
    )

    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/portfolio_{port}.pdf")
