"""
ML utilities
"""

import numpy as np
import tensorflow as tf
import torch
from scipy.sparse import spmatrix


def ensemble_prediction(
    models: list[tf.keras.models.Sequential], x: np.ndarray | spmatrix
) -> list[float]:
    """
    Function to make ensemble prediction
    """

    prediction_list = np.mean(
        np.array(
            [model.predict(x, verbose=0, batch_size=10000).ravel() for model in models]
        ),
        axis=0,
    )

    return list(prediction_list)

def mse_loss(y, y_pred):
    """
    Mean square error loss
    """
    return torch.mean((y - y_pred) ** 2)

def huber_loss(y, y_pred, sigma=0.1):
    """
    Huber loss
    """
    return torch.mean(
        torch.where(
            torch.abs(y - y_pred) <= sigma,
            (y - y_pred) ** 2,
            2 * sigma * (torch.abs(y - y_pred) - sigma),
        )
    )

def l1_penalty(lamb, b):
    """
    L1 penalty
    """
    return lamb * torch.sum(torch.abs(b))

def l2_penalty(lamb, b):
    """
    L2 penalty
    """
    return lamb * torch.sum(b ** 2)

def elastic_penalty(lamb, b, alpha):
    """
    Elastic net penalty
    """
    return alpha * l1_penalty(lamb, b) + (1 - alpha) * l2_penalty(lamb, b)