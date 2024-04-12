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
ML_METHOD = ["rf"] + [f"nn_{_}" for _ in range(1, 6)]

ML_NAMING_DICT = {
    "rf" : "RF",
    "nn_1": "NN1",
    "nn_2": "NN2",
    "nn_3": "NN3",
    "nn_4": "NN4",
    "nn_5": "NN5",
}

# hyperparameters
PARAM_GRID = {
    "nn": {
        "hidden_layer_sizes": [
            (128,),
            (128, 64),
            (128, 64, 32),
            (128, 64, 32, 16),
            (128, 64, 32, 16, 8),
        ],
        "alpha": [10**_ for _ in np.linspace(-4, -3, num=5)],
        "learning_rate_init": np.linspace(0.001, 0.1, num=5),
    },
    "rf":{
        "max_depth": range(1, 7, 1),
        "max_features": [3, 5, 10, 20],
    }
}
PARAM_GRID

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
