import os
import sys

import dotenv
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import NeptuneLogger
from src.data import DataConfig, get_data_loader

utils_package_path = os.path.abspath(os.path.join(".", "..", ".."))
sys.path.append(utils_package_path)

from utils.evaluation import evaluate_lightning_module


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(config: DictConfig):

    pl.seed_everything(42, workers=True)
    dotenv.load_dotenv(config.environment_file)

    neptune_logger = NeptuneLogger(
        name=config.experiment.name,
        project=config.neptune_project,
    )

    model = hydra.utils.instantiate(config.experiment.model)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=neptune_logger,
        log_every_n_steps=config.log_every_n_steps,
        accelerator=config.accelerator,
        deterministic=True,
    )
    data_config = DataConfig(
        config.experiment.batch_size,
        config.num_workers,
        config.shuffle_train_split,
    )
    train_loader = get_data_loader("train", data_config, logger=neptune_logger)
    val_loader = get_data_loader(
        "val",
        data_config,
        logger=neptune_logger,
    )

    trainer.fit(model, train_loader, val_loader)

    evaluate_lightning_module(model, neptune_logger)

    neptune_logger.log_model_summary(model=model, max_depth=-1)


if __name__ == "__main__":
    main()
