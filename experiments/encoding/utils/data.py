import os
import pathlib

import pandas as pd
import torch

DTYPE = torch.float32


def _load_data(path: str, dtype: torch.dtype = DTYPE):
    X_path = os.path.join(path, "X_pandas.pck")
    y_path = os.path.join(path, "y_pandas.pck")

    X: pd.DataFrame = pd.read_pickle(X_path)
    y: pd.DataFrame = pd.read_pickle(y_path)
    return torch.tensor(X.to_numpy()).to(dtype), torch.tensor(y.to_numpy()).to(dtype)


def get_data(path: str):
    return _load_data(
        os.path.abspath(
            os.path.join(pathlib.Path(__file__).parent.resolve(), "../../..", path)
        )
    )
