import os
import pathlib
import typing

import pandas as pd
import torch

D_TYPE = torch.float32

ValidSplit = typing.Literal["train", "test", "val"]


def convert_to_pandas(X: torch.Tensor, y: torch.Tensor):
    raise NotImplementedError()


def convert_to_torch(df: pd.DataFrame):
    return torch.tensor(df.values)


def _load_data(path: str):
    X_path = os.path.join(path, "X_pandas.pck")
    y_path = os.path.join(path, "y_pandas.pck")

    X: pd.DataFrame = pd.read_pickle(X_path)
    y: pd.DataFrame = pd.read_pickle(y_path)
    return convert_to_torch(X).to(D_TYPE), convert_to_torch(y).to(D_TYPE)


def generate_log_data(dataset):
    def format_size(size):
        return "x".join(map(str, size))

    return {
        key: str(value)
        for key, value in {
            "no_samples": len(dataset),
            "no_input_features": format_size(dataset[0][0].shape),
            "no_output_features": format_size(dataset[0][1].shape),
        }.items()
    }


class DataConfig(typing.NamedTuple):
    batch_size: int
    num_workers: int
    shuffle_train_split: bool
    persistent_workers: bool


def get_data_loader(
    split: ValidSplit,
    config: DataConfig,
    logger,
    x_only: bool = False,
    transform_x=None,
):
    if split not in typing.get_args(ValidSplit):
        raise ValueError("Split not available.")

    X, y = _load_data(
        os.path.abspath(
            os.path.join(
                pathlib.Path(__file__).parent.resolve(),
                "../../..",
                "data",
                "splits",
                split,
            )
        )
    )

    if transform_x:
        X = transform_x(X)

    if x_only:
        dataset = torch.utils.data.TensorDataset(X)
    else:
        dataset = torch.utils.data.TensorDataset(X, y)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        config.batch_size,
        persistent_workers=config.persistent_workers,
        shuffle=config.shuffle_train_split and split == "train",
        num_workers=config.num_workers,
    )
    # logger.log_hyperparams(metrics=generate_log_data(dataset))

    # With NEPTUNE LOGGER:

    # for key, value in generate_log_data(dataset).items():
    #    logger.experiment[f"data/{split}/info/{key}"].log(value)
    # logger.experiment[f"data/{split}/config"].log(str(config))

    return data_loader
