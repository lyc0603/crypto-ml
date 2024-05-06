"""
Function to run the neural network
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import torch
from scipy.sparse import spmatrix
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from environ.constants import VALIDATION_MONTH
from environ.process.ml_utils import (elastic_penalty, huber_loss, l1_penalty,
                                      l2_penalty, mse_loss)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def transform_train(
    df: pd.DataFrame, var: list
) -> tuple[pd.Series, np.ndarray | spmatrix, preprocessing.StandardScaler]:
    """
    Function to transform the data using the standard scaler
    """

    ind_var = df[[_ for _ in var]].copy()
    d_var = df["log_eret_w"]
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
    d_var = df["log_eret_w"]
    ind_var = scaler.transform(ind_var)
    return d_var, ind_var


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Function to calculate the r2 score
    """
    return 1 - sum((y_true - y_pred) ** 2) / sum((y_true) ** 2)


def sample_split(
    df: pd.DataFrame,
    time: pd.Timestamp,
    xvar: list,
    mvar: list,
    cvar: list,
    dvar: str = "",
) -> tuple[
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
        if dvar == "category":
            for c in cvar:
                df[c] = 0
        else:
            df[dvar] = 0

    # create the interaction variables
    ivar = []
    for x in xvar:
        for m in mvar:
            var_name = f"{x}_{m}"
            df[var_name] = df[x] * df[m]
            ivar.append(var_name)

    var = xvar + ivar + cvar

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


class LinearModel(torch.nn.Module):
    """
    Class for linear model
    """

    def __init__(
        self, lamb=0.1, lr=1e-5, penalty=None, n_iter=10000, fit_intercept=True, alpha=0.5
    ):
        super().__init__()
        self.history = []
        self.lamb = lamb
        self.lr = lr
        self.penalty = penalty
        self.n_iter = n_iter
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        return

    def add_intercept(self, x):
        """
        Function to add intercept
        """
        a = torch.ones(x.size(0), 1)
        return torch.cat([a, x], 1)

    def forward(self, x):
        """
        Linear model
        """
        return x @ self.b.t() + self.const

    def loss(self, y, y_pred, loss_func="huber", sigma=1.35):
        """
        Calculate the loss
        """

        rss = huber_loss(y, y_pred, sigma) if loss_func == "huber" else mse_loss(y, y_pred)

        penalty = 0
        if self.penalty == "l1":
            penalty = l1_penalty(self.lamb, self.b)
        if self.penalty == "l2":
            penalty = l2_penalty(self.lamb, self.b)
        if self.penalty == "enet":
            penalty = elastic_penalty(self.lamb, self.b, self.alpha)

        return rss + penalty


    def fit(self, x, y, loss_func="huber", sigma=1.35):
        """
        Function to fit the model
        """

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        if self.fit_intercept:
            x = self.add_intercept(x)

        self.b = torch.nn.Parameter(torch.zeros(x.shape[1]))
        self.const = torch.nn.Parameter(torch.zeros(1))

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for _ in range(self.n_iter):
            x_ = torch.FloatTensor(x)
            y_ = torch.FloatTensor(y)

            y_pred = self.forward(x_)
            optimizer.zero_grad()
            loss = self.loss(y_, y_pred, loss_func, sigma)
            loss.backward()
            optimizer.step()

            self.history.append(loss.item())

        return self.b.detach().numpy(), self.const.detach().numpy()

    def predict(self, x):
        """
        Function to predict
        """
        x = torch.FloatTensor(x)
        if self.fit_intercept:
            x = self.add_intercept(x)
        return self.forward(x).detach().numpy()

    def plot_history(self):
        """
        Function to plot the history
        """
        return sns.lineplot(
            x=[i + 1 for i in range(len(self.history))], y=self.history
        ).set(xlabel="Iteration", ylabel="Loss", title="History")

def pls_model(
    x_train: np.ndarray | spmatrix,
    y_train: pd.Series,
    n_components: int,
    ) -> Any:
    """
    Function to create the PLS model
    """
    
    model = PLSRegression(n_components=n_components)
    model.fit(x_train, y_train)
    
    return model

def pcr_model(
    x_train: np.ndarray | spmatrix,
    y_train: pd.Series,
    n_components: int,
    ) -> Any:
    """
    Function to create the PCR model
    """

    model = make_pipeline(PCA(n_components=n_components), LinearRegression())
    model.fit(x_train, y_train)

    return model



def gbrt_model(
    x_train: np.ndarray | spmatrix,
    y_train: pd.Series,
    max_depth: int,
    tree_num: int,
    learning_rate: float,
) -> Any:
    """
    Function to create the gradient boosting model
    """
    
    model = GradientBoostingRegressor(
        max_depth=max_depth,
        n_estimators=tree_num,
        learning_rate=learning_rate,
        random_state=0,
        loss="huber",
    )

    model.fit(x_train, y_train)
    return model



def rf_model(
    x_train: np.ndarray | spmatrix,
    y_train: pd.Series,
    max_depth: int,
    max_features: int,
) -> Any:
    """
    Function to create the random forest model
    """

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
        random_state=0,
    )

    model.fit(x_train, y_train)
    return model


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
