"""
Function to run the neural network
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.sparse import spmatrix
from sklearn import preprocessing

from environ.constants import VALIDATION_MONTH

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def transform_train(
    df: pd.DataFrame, var: list
) -> tuple[pd.Series, np.ndarray | spmatrix, preprocessing.StandardScaler]:
    """
    Function to transform the data using the standard scaler
    """

    ind_var = df[[_ for _ in var]].copy()
    d_var = df["ret_w"]
    scaler = preprocessing.StandardScaler().fit(ind_var)
    ind_var = scaler.transform(ind_var)
    return d_var, ind_var, scaler


def gen_data(
    df: pd.DataFrame, var: list, scaler: Any
) -> tuple[pd.Series, np.ndarray | spmatrix]:
    """
    Generate the data
    """
    ind_var = df[[_ for _ in var]].copy()
    d_var = df["ret_w"]
    ind_var = scaler.transform(ind_var)
    return d_var, ind_var


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Function to calculate the r2 score
    """
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true) ** 2)


def sample_split(df: pd.DataFrame, time: pd.Timestamp, var: list, dvar: str = "") -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    np.ndarray | spmatrix,
    pd.Series,
    np.ndarray | spmatrix,
    pd.Series,
    np.ndarray | spmatrix,
]:
    """
    Function to split the data
    """

    # set the dvar to zero if provided
    if dvar:
        df[dvar] = 0
        print(f"{dvar} set to zero")
    else:
        print("No dvar provided")

    train = df.loc[df["time"] < time - pd.DateOffset(months=VALIDATION_MONTH)]
    valid = df.loc[
        (df["time"] >= time - pd.DateOffset(months=VALIDATION_MONTH))
        & (df["time"] < time)
    ]
    test = df.loc[(df["time"] == time)]

    y_train, x_train, scaler = transform_train(train, var)
    y_valid, x_valid = gen_data(valid, var, scaler)
    y_test, x_test = gen_data(test, var, scaler)

    return (
        train,
        valid,
        test,
        y_train,
        x_train,
        y_valid,
        x_valid,
        y_test,
        x_test,
    )


def nn_model(
    x_train: np.ndarray | spmatrix,
    y_train: pd.Series,
    hidden_layer_sizes,
    alpha,
    learning_rate_init,
) -> tf.keras.models.Sequential:
    """
    Function to create the neural network model
    """

    # input layer
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                hidden_layer_sizes[0],
                activation="relu",
                input_shape=(x_train.shape[1],),
                kernel_regularizer=tf.keras.regularizers.l1(alpha),
            ),
            tf.keras.layers.BatchNormalization(),
        ]
    )

    # hidden layers
    for layer in hidden_layer_sizes[1:]:
        model.add(
            tf.keras.layers.Dense(
                layer,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l1(alpha),
            )
        )
        model.add(tf.keras.layers.BatchNormalization())

    # output layer
    model.add(tf.keras.layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_init),
        loss="mean_squared_error",
    )

    # fit the model
    model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=10000,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=5, restore_best_weights=True
            )
        ],
    )
    return model
