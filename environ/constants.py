"""
Constants for the project
"""

from pathlib import Path

import numpy as np

from environ.settings import PROJECT_ROOT

# Paths
DATA_PATH: Path = PROJECT_ROOT / "data"
PROCESSED_DATA_PATH: Path = PROJECT_ROOT / "processed_data"
TABLE_PATH: Path = PROJECT_ROOT / "tables"
FIGURE_PATH: Path = PROJECT_ROOT / "figures"

# Coingecko API KEY
COINGECKO_API_KEY = [
    "CG-p19fJQK2mNHmCaZsjAXPSndt",
    "CG-MsksHi64zG3pWyguYpKAGhEi",
    "CG-2KLZH7JsRS8TDa2snEcCpMZA",
    "CG-uN4cmEUSWZsuidix2827hk3g",
    "CG-8wYJUwkBbQ5WaDcVZTKUe2kb",
]

# sample split
VALIDATION_MONTH = 1
TEST_START_DATE = "2021-01-01"

# ml method
ML_METHOD = ["ols", "lasso", "enet", "rf"] + [f"nn_{_}" for _ in range(1, 6)]

ML_NAMING_DICT = {
    "ols": "OLS+H",
    "lasso": "LASSO+H",
    "enet": "ENet+H",
    "rf": "RF",
    "nn_1": "NN1",
    "nn_2": "NN2",
    "nn_3": "NN3",
    "nn_4": "NN4",
    "nn_5": "NN5",
}

ML_PATH = {
    "ols": "res",
    "lasso": "res_comp",
    "enet": "res_comp",
    "rf": "res",
    "nn_1": "res",
    "nn_2": "res",
    "nn_3": "res",
    "nn_4": "res",
    "nn_5": "res",
}

# hyperparameters
PARAM_GRID = {
    "pls": {
        "n_components": [1, 3, 5, 10, 20, 50],
    },
    "enet": {
        "lamb": [10**_ for _ in np.linspace(-4, -1, num=5)],
    },
    "lasso": {
        "lamb": [10**_ for _ in np.linspace(-4, -1, num=5)],
    },
    "gbrt": {
        "max_depth": [1, 2],
        "tree_num": [250, 500, 750, 1000],
        "learning_rate": [0.01, 0.1],
    },
    "rf": {
        "max_depth": range(1, 7, 1),
        "max_features": [3, 5, 10, 20, 30, 50],
    },
    "nn": {
        "hidden_layer_sizes": [
            # (128,),
            # (128, 64),
            # (128, 64, 32),
            # (128, 64, 32, 16),
            (128, 64, 32, 16, 8),
        ],
        "alpha": [10**_ for _ in np.linspace(-5, -3, num=5)],
        "learning_rate_init": [0.001, 0.01],
    },
}

CATEGORY_LIST = [
    "layer-1",
    "layer-2",
    "smart-contract-platform",
    "alleged-sec-securities",
    "exchange-based-tokens",
    "centralized-exchange-token-cex",
    "decentralized-exchange",
    "decentralized-finance-defi",
    "meme-token",
    "governance",
]

# risk-free rate
FAMA_FRENCH_DAILY_FACTOR = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    + "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)
User_Agent = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        + "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
