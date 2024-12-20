import os
from typing import Callable, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import NeptuneLogger

from utils.path import get_test_data_path

from .metrics import Metrics


def evaluate(
    make_prediction: Callable[[pd.DataFrame], pd.DataFrame],
):
    """
    Evaluate the performance of a make_prediction function on input data and logs metrics.

    The data used is defined in the data directory under splits/test

    This function accepts a make_prediction function, which is a callable that transforms an input
    DataFrame (X) into an output DataFrame (y). It evaluates the model's performance
    using predefined metrics.

    Parameters:
    ----------
    make_prediction : Callable[[pd.DataFrame], pd.DataFrame]
        A callable function or object that takes an input DataFrame `X`
        and returns an output DataFrame `y`, representing a model's predictions.
    Returns:
    -------
    metrics : Metrics

    Example:
    -------
    >>> def evaluate_from_dataframe(X: pd.DataFrame):
    >>>     X_tensor = torch.tensor(X.to_numpy())
    >>>
    >>>     #model: a pytorch model, which transforms X -> y in torch.Tensor format
    >>>     y_pred_tensor = model(X_tensor)
    >>>
    >>>     return pd.DataFrame(y_pred_tensor)
    >>>
    >>> evaluate(evaluation_function)
    """
    X_test = pd.read_pickle(os.path.join(get_test_data_path(), "X_pandas.pck"))
    y_test = pd.read_pickle(os.path.join(get_test_data_path(), "y_pandas.pck"))

    y_pred = make_prediction(X_test)

    metrics = Metrics(y_pred.to_numpy(), y_test.to_numpy())

    return metrics


def evaluate_lightning_module(model: pl.LightningModule, neptune_logger: NeptuneLogger):
    def evaluate_from_dataframe(X: pd.DataFrame):
        X_tensor = torch.tensor(X.to_numpy())
        y_pred_tensor = model(X_tensor)
        return pd.DataFrame(y_pred_tensor.detach().cpu().to(torch.int32))

    metrics = evaluate(evaluate_from_dataframe)

    for key, value in metrics:
        neptune_logger.experiment[key] = value
