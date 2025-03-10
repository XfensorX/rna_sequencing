import os
import sys

import dotenv
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from src.data import DataConfig, get_data_loader

utils_package_path = os.path.abspath(os.path.join(".", "..", ".."))
sys.path.append(utils_package_path)

from utils.evaluation import evaluate_lightning_module


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.3")
def main(config: DictConfig):

    def x_transform(x: torch.Tensor):
        if "shift_input" in config.experiment and config.experiment.shift_input:
            return x - 0.5

        return x

    pl.seed_everything(42, workers=True)
    dotenv.load_dotenv(config.environment_file)

    neptune_logger = TensorBoardLogger(
        save_dir="runs",
        name=config.experiment.name,
        log_graph=True,
    )

    model = hydra.utils.instantiate(config.experiment.model)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=neptune_logger,
        log_every_n_steps=config.log_every_n_steps,
        accelerator=config.accelerator,
        deterministic=True,
        default_root_dir="lightning",
    )

    data_config = DataConfig(
        config.experiment.batch_size,
        config.num_workers,
        config.shuffle_train_split,
        config.persistent_workers,
    )
    train_loader = get_data_loader(
        "train",
        data_config,
        logger=neptune_logger,
        x_only=config.experiment.x_only,
        transform_x=x_transform,
    )
    val_loader = get_data_loader(
        "val",
        data_config,
        logger=neptune_logger,
        x_only=config.experiment.x_only,
        transform_x=x_transform,
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    finally:
        metrics = evaluate_lightning_module(model, x_transform=x_transform)
        neptune_logger.log_hyperparams(dict(metrics))
        neptune_logger.finalize("success")
        neptune_logger.save()

    # NEPTUNE LOGGER:
    # for key, value in metrics:
    #   neptune_logger.experiment[f"evaluation/metrics/{key}"] = value
    # neptune_logger.log_model_summary(model=model, max_depth=-1)


if __name__ == "__main__":
    main()
