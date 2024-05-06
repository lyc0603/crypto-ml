"""
Function to run the neural network
"""

import os

import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression

from environ.constants import PARAM_GRID
from environ.process.ml_model import (
    LinearModel,
    nn_model,
    r2_score,
    rf_model,
    gbrt_model,
    pls_model,
    pcr_model,
    sample_split,
)
from environ.process.ml_utils import ensemble_prediction

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def ols(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], RandomForestRegressor, list[dict[str, float]]]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    model = LinearModel(penalty=None, lr=1e-5, n_iter=10000)
    model.fit(x_train, y_train, loss_func="huber")
    s_valid = r2_score(y_valid, model.predict(x_valid))
    res.append((s_valid, model))

    s_valid, opt_model = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        opt_model,
        params_list,
    )


def pcr(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], PLSRegression, list[dict[str, float]]]:
    """
    Function to run the PCR
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    for k in PARAM_GRID["pcr"]["n_components"]:
        model = pcr_model(x_train, y_train, n_components=k)
        s_valid = r2_score(y_valid, model.predict(x_valid))
        res.append((s_valid, model, k))

    s_valid, opt_model, k = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "n_components": k,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        opt_model,
        params_list,
    )


def pls(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], PLSRegression, list[dict[str, float]]]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    for k in PARAM_GRID["pls"]["n_components"]:
        model = pls_model(x_train, y_train, n_components=k)
        s_valid = r2_score(y_valid, model.predict(x_valid))
        res.append((s_valid, model, k))

    s_valid, opt_model, k = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "n_components": k,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        opt_model,
        params_list,
    )


def lasso(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], RandomForestRegressor, list[dict[str, float]]]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    for lamb in PARAM_GRID["lasso"]["lamb"]:
        model = LinearModel(penalty="l1", lr=1e-5, n_iter=10000, lamb=lamb)
        model.fit(x_train, y_train, loss_func="huber")
        s_valid = r2_score(y_valid, model.predict(x_valid))
        res.append((s_valid, model, lamb))

    s_valid, opt_model, lamb = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "lamb": lamb,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        opt_model,
        params_list,
    )


def enet(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], RandomForestRegressor, list[dict[str, float]]]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    for lamb in PARAM_GRID["enet"]["lamb"]:
        model = LinearModel(penalty="enet", lr=1e-5, n_iter=10000, lamb=lamb, alpha=0.5)
        model.fit(x_train, y_train, loss_func="huber")
        s_valid = r2_score(y_valid, model.predict(x_valid))
        res.append((s_valid, model, lamb))

    s_valid, opt_model, lamb = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "lamb": lamb,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        opt_model,
        params_list,
    )


def gbrt(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], GradientBoostingRegressor, list[dict[str, float]]]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    for max_depth in PARAM_GRID["gbrt"]["max_depth"]:
        for tree_num in PARAM_GRID["gbrt"]["tree_num"]:
            for learning_rate in PARAM_GRID["gbrt"]["learning_rate"]:
                model = gbrt_model(x_train, y_train, max_depth, tree_num, learning_rate)
                s_valid = r2_score(y_valid, model.predict(x_valid))
                res.append((s_valid, model, max_depth, tree_num, learning_rate))
                print(s_valid, max_depth, tree_num, learning_rate)

    s_valid, opt_model, opt_max_depth, opt_tree_num, opt_learning_rate = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "max_depth": opt_max_depth,
                "tree_num": opt_tree_num,
                "learning_rate": opt_learning_rate,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")
    return (
        test_dict,
        opt_model,
        params_list,
    )


def rf(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
) -> tuple[dict[str, pd.DataFrame], RandomForestRegressor, list[dict[str, float]]]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    for max_depth in PARAM_GRID["rf"]["max_depth"]:
        for max_features in PARAM_GRID["rf"]["max_features"]:
            model = rf_model(x_train, y_train, max_depth, max_features)
            s_valid = r2_score(y_valid, model.predict(x_valid))
            res.append((s_valid, model, max_depth, max_features))

    s_valid, opt_model, opt_max_depth, opt_max_features = max(res)

    # var importance
    test_dict = {}
    params_list = []

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        train["log_eret_pred"] = opt_model.predict(x_train)
        test["log_eret_pred"] = opt_model.predict(x_test)
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "max_depth": opt_max_depth,
                "opt_features": opt_max_features,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        opt_model,
        params_list,
    )


def nn(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list[str],
    mvar: list[str],
    cvar: list[str],
    hidden_layer_sizes: tuple[int],
) -> tuple[
    dict[str, pd.DataFrame], list[tf.keras.models.Sequential], list[dict[str, float]]
]:
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
    ) = sample_split(dff, time, xvar, mvar, cvar)

    res = []
    model_list = []
    for alpha in PARAM_GRID["nn"]["alpha"]:
        for learning_rate_init in PARAM_GRID["nn"]["learning_rate_init"]:
            model = nn_model(
                x_train, y_train, hidden_layer_sizes, alpha, learning_rate_init
            )
            s_valid = r2_score(y_valid, model.predict(x_valid, verbose=0).ravel())
            res.append((s_valid, model, alpha, learning_rate_init))
            print(
                f"alpha: {alpha}, learning_rate_init: {learning_rate_init}, valid_score: {s_valid}"
            )

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

    for dvar in [""] + xvar + mvar + ["category"]:
        dfv = df.copy()
        (
            train,
            _,
            test,
            y_train,
            x_train,
            _,
            _,
            _,
            x_test,
        ) = sample_split(dfv, time, xvar, mvar, cvar, dvar)
        test["log_eret_pred"] = ensemble_prediction(model_list, x_test)
        train["log_eret_pred"] = ensemble_prediction(model_list, x_train)
        s_test = r2_score(test["log_eret_w"], test["log_eret_pred"])
        s_train = r2_score(train["log_eret_w"], train["log_eret_pred"])

        # save the prediction, params, and scores
        test_dict[dvar] = test.copy()
        params_list.append(
            {
                "date": time,
                "dvar": dvar,
                "alpha": opt_alpha,
                "learning_rate_init": opt_learning_rate_init,
                "train_score": s_train,
                "valid_score": s_valid,
                "test_score": s_test,
            }
        )
        print(f"dvar: {dvar}, train_score: {s_train}, test_score: {s_test}")

    return (
        test_dict,
        model_list,
        params_list,
    )
