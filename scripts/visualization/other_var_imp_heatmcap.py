"""
Script to visualize the importance of macro variables in predicting the target variable.
"""

from scripts.process.other_var_imp import factor_importance_df
import matplotlib.pyplot as plt
import seaborn as sns

from environ.constants import FIGURE_PATH

plt.rcParams["figure.figsize"] = (12, 6)

sns.heatmap(factor_importance_df, cmap="Blues", cbar=False)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}/other_var_imp.pdf")
plt.show()
