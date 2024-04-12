"""
Script to visualize the factor importance of the model
"""

import matplotlib.pyplot as plt
import seaborn as sns

from environ.constants import FIGURE_PATH, ML_METHOD
from scripts.process.factor_importance import factor_importance_dict

plt.rcParams["figure.figsize"] = (8, 5)

for method in ML_METHOD:
    df_param_importance = factor_importance_dict[method]
    sns.heatmap(df_param_importance, cmap="Blues", cbar=False)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}/fi_{method}.pdf")
