"""
Function to run the neural network
"""

import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestRegressor

from environ.constants import PARAM_GRID
from environ.process.ml_model import nn_model, r2_score, rf_model, sample_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ensemble_prediction(
    models: list[tf.keras.models.Sequential], x: np.ndarray | spmatrix
) -> list[float]:
    """
    Function to make ensemble prediction
    """

    prediction_list = np.mean(
        np.array([model.predict(x, verbose=0).ravel() for model in models]), axis=0
    )

    return list(prediction_list)

def rf(
    df: pd.DataFrame, time: pd.Timestamp, var: list[str]
) -> tuple[dict[str, pd.DataFrame],RandomForestRegressor, list[dict[str, float]]]:
    """
    Function to run the neural network
    """

    dff = df.copy()
    # split the data
    (
        _,
        _,
        test,
        y_train,
        x_train,
        y_valid,
        x_valid,
        _,
        x_test,
    ) = sample_split(dff, time, var)

    res = []
    for max_depth in PARAM_GRID["rf"]["max_depth"]:
        for max_features in PARAM_GRID["rf"]["max_features"]:
            model = rf_model(
                x_train, y_train, max_depth, max_features
            )
            s_valid = r2_score(y_valid, model.predict(x_valid))
            res.append((s_valid, model, max_depth, max_features))

    s_valid, opt_model, opt_max_depth, opt_max_features = max(res)

    # ensemble and var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + var:
        dfv = df.copy()
        (
            _,
            _,
            test,
            _,
            _,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, var, dvar)
        test["ret_pred"] = model.predict(x_test)
        s_test = r2_score(test["ret_w"], test["ret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
        {
            "date": time,
            "dvar": dvar,
            "max_depth": opt_max_depth,
            "opt_features": opt_max_features,
            "valid_score": s_valid,
            "test_score": s_test,
        }
        )

    return (
        test_dict,
        opt_model,
        params_list,
    )

def nn(
    df: pd.DataFrame, time: pd.Timestamp, var: list[str], hidden_layer_sizes: tuple[int]
) -> tuple[dict[str, pd.DataFrame], list[tf.keras.models.Sequential], list[dict[str, float]]]:
    """
    Function to run the neural network
    """

    dff = df.copy()
    # split the data
    (
        _,
        _,
        test,
        y_train,
        x_train,
        y_valid,
        x_valid,
        _,
        x_test,
    ) = sample_split(dff, time, var)

    res = []
    model_list = []
    for alpha in PARAM_GRID["nn"]["alpha"]:
        for learning_rate_init in PARAM_GRID["nn"]["learning_rate_init"]:
            model = nn_model(
                x_train, y_train, hidden_layer_sizes, alpha, learning_rate_init
            )
            s_valid = r2_score(y_valid, model.predict(x_valid, verbose=0).ravel())
            res.append((s_valid, model, alpha, learning_rate_init))

    s_valid, opt_model, opt_alpha, opt_learning_rate_init = max(res)

    # ensemble and var importance
    model_list.append(opt_model)
    for _ in range(10 - 1):
        model = nn_model(
            x_train, y_train, hidden_layer_sizes, opt_alpha, opt_learning_rate_init
        )
        model_list.append(model)
    test_dict = {}
    params_list = []

    for dvar in [""] + var:
        dfv = df.copy()
        (
            _,
            _,
            test,
            _,
            _,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, var, dvar)
        test["ret_pred"] = ensemble_prediction(model_list, x_test)
        s_test = r2_score(test["ret_w"], test["ret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "alpha": opt_alpha,
                "learning_rate_init": opt_learning_rate_init,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )

    return (
        test_dict,
        model_list,
        params_list,
    )
