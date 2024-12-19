import os
import sys

import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger
from src.data import get_data_loader

utils_package_path = os.path.abspath(os.path.join(".", "..", ".."))
sys.path.append(utils_package_path)

from utils.evaluation import evaluate_lightning_module


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(config: DictConfig):

    pl.seed_everything(42, workers=True)
    dotenv.load_dotenv(config.framework.environment)

    neptune_logger = NeptuneLogger(
        name=config.experiment.name,
        project=config.neptune.project,
    )

    model = hydra.utils.instantiate(config.experiment.model)

    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        logger=neptune_logger,
        log_every_n_steps=config.training.log_every_n_steps,
        accelerator=config.framework.accelerator,
        deterministic=True,
    )

    train_loader = get_data_loader("train", config.data, logger=neptune_logger)
    val_loader = get_data_loader("val", config.data, logger=neptune_logger)

    trainer.fit(model, train_loader, val_loader)

    evaluate_lightning_module(model, neptune_logger)

    neptune_logger.log_model_summary(model=model, max_depth=-1)


if __name__ == "__main__":
    main()
