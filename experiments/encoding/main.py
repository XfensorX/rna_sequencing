import os
import sys

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger
from src.data import get_data_loader
import dotenv

utils_package_path = os.path.abspath(os.path.join(".", "..", ".."))
sys.path.append(utils_package_path)

from utils import logging, evaluation


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(config: DictConfig):
    dotenv.load_dotenv(config.environment)

    neptune_logger = NeptuneLogger(
        name=config.neptune.name,
        project=config.neptune.project,
    )

    model = hydra.utils.instantiate(config.model)

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=neptune_logger,
        log_every_n_steps=config.training.log_every_n_steps,
        accelerator=config.accelerator,
    )

    train_loader = get_data_loader("train", config.data, logger=neptune_logger)

    trainer.fit(model, train_loader)

    def eval_model(model: pl.LightningModule, neptune_logger: NeptuneLogger):
        def evaluate_from_dataframe(X: pd.DataFrame):
            X_tensor = torch.tensor(X.to_numpy())
            y_pred_tensor = model(X_tensor)
            return pd.DataFrame(y_pred_tensor.detach().cpu().to(torch.int32))

        evaluation.evaluate(
            evaluate_from_dataframe,
            logging.Logger.from_lightning_neptune_logger(neptune_logger, "evaluation"),
        )

    eval_model(model, neptune_logger)


if __name__ == "__main__":
    main()
