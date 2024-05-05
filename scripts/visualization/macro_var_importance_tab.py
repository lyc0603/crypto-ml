"""
Script to generate a table of the importance of macro variables in the model
"""

from scripts.process.macro_var_imp import factor_importance_dict
from environ.constants import ML_METHOD, TABLE_PATH, H_ML, MACRO_VARIABLES

# generate latex table
with open(f"{TABLE_PATH}/macro_var_imp.tex", "w", encoding="utf-8") as f:
    f.write(
        r"\begin{tabularx}{\linewidth}{*{"+ f"{len(factor_importance_dict.keys())+ 1}" +r"}{X}}"
    )
    f.write("\n")
    f.write(r"\toprule")
    f.write("\n")
    for ml_method in factor_importance_dict.keys():
        f.write(f"& {ml_method}")
    f.write(r"\\")
    f.write("\n")
    for ml_method in factor_importance_dict.keys():
        f.write(f"& +H") if ml_method in H_ML else f.write(f"& ")
    f.write(r"\\")
    f.write("\n")
    f.write(r"\midrule")
    f.write("\n")
    for macro_var in MACRO_VARIABLES:
        f.write(f"{macro_var}")
        for ml_method, ml_df in factor_importance_dict.items():
            facotr_importance = ml_df.loc[ml_df["dvar"] == macro_var, 'train_score'].values[0]
            f.write(f"& {facotr_importance}")
        f.write(r"\\")
        f.write("\n")
    f.write(r"\bottomrule")
    f.write("\n")
    f.write("\end{tabularx}")