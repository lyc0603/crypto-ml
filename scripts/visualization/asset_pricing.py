"""
Script to visualize asset pricing table
"""

from environ.constants import TABLE_PATH, ML_NAMING_DICT
from scripts.process.asset_pricing import ML_METHOD, asset_pricing_dict

# generate latex table
with open(f"{TABLE_PATH}/asset_pricing.tex", "w", encoding="utf-8") as f:
    f.write(
        r"\begin{tabular}{ccccccccccccc}"
    )
    f.write("\n")
    for idx in range(1, len(ML_METHOD) + 1, 3):
        for idx_row in range(idx, idx + 3):
            f.write(
                f" & \\multicolumn{{4}}{{c}}{{{ML_NAMING_DICT[ML_METHOD[idx_row - 1]]}}}"
            )
        f.write(r"\\")
        f.write("\n")
        f.write("\hline\n")
        for idx_row in range(idx, idx + 3):
            f.write("& Avg & Pred & SD & SR ")

        f.write(r"\\")
        f.write("\n")
        f.write("\hline\n")
        for quantile in range(1, 6, 1):
            if quantile == 1:
                f.write("Low(L)")
            elif quantile == 5:
                f.write("High(H)")
            else:
                f.write(f"{quantile}")
            for idx_row in range(idx, idx + 3):
                for metric in ["Avg", "Pred", "SD", "SR"]:
                    f.write(f"& {asset_pricing_dict[ML_METHOD[idx_row - 1]][quantile][metric]}")
            f.write(r"\\")
            f.write("\n")
        f.write("\hline\n")

    f.write("\end{tabular}")
    



