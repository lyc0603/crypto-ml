"""
Script to generate table for R2 score
"""

from scripts.process.r2 import r2_dict
from environ.constants import TABLE_PATH

# generate latex table
with open(f"{TABLE_PATH}/r2.tex", "w", encoding="utf-8") as f:
    f.write(
        r"\begin{tabularx}{\linewidth}{*{" + f"{len(r2_dict.keys())}" + r"}{X}}"
    )
    f.write("\n")
    for ml_method in r2_dict.keys():
        f.write(f"& {ml_method}")
    f.write(r"\\")
    f.write("\n")
    f.write("\hline\n")
    f.write(r"All")
    for ml_method in r2_dict.keys():
        f.write(f"& {r2_dict[ml_method]['All']}")
    f.write(r"\\")
    f.write("\n")
    f.write("\hline\n")
    f.write(r"Top 10\%")
    for ml_method in r2_dict.keys():
        f.write(f"& {r2_dict[ml_method]['Top']}")
    f.write(r"\\")
    f.write("\n")
    f.write("\hline\n")
    f.write(r"Bottom 10\%")
    for ml_method in r2_dict.keys():
        f.write(f"& {r2_dict[ml_method]['Bottom']}")
    f.write(r"\\")
    f.write("\n")
    f.write("\hline\n")
    f.write("\end{tabularx}")