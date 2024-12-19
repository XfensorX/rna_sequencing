import os
from typing import Any, Callable

import neptune
from pytorch_lightning.loggers import NeptuneLogger

from .metrics import Metrics

LogMethod = Callable[[str, Any], None]


class Logger:
    """
    A Logger class that provides functionality to log a combination of names and values.

    The Logger class can be instantiated with various log methods and provides
    an easy way to log metrics. It includes factory methods to create Logger
    instances for different logging backends, such as Neptune logger and standard output.
    """

    def __init__(self, log_method: LogMethod):

        self.log = log_method

    def log_metrics(self, metrics: Metrics):
        for name, value in metrics:
            self.log(f"metrics/{name}", value)

    @classmethod
    def from_lightning_neptune_logger(
        cls, neptune_logger: NeptuneLogger, logging_path: str
    ) -> "Logger":
        """
        Args:
            neptune_logger (NeptuneLogger): The Lightning Neptune logger instance.
            logging_path (str): The path where logs should be stored in the Neptune experiment.
        """

        def log(name: str, value: Any):
            neptune_logger.experiment[str(os.path.join(logging_path, name))] = value

        return Logger(log)

    @classmethod
    def for_neptune(cls, run: neptune.Run, logging_path: str) -> "Logger":
        """
        Args:
            run (neptune.Run): A Neptune Run instance to use for logging.
            logging_path (str): The path where logs should be stored in the Neptune experiment.
        """

        def log(name: str, value: Any):
            run[str(os.path.join(logging_path, name))].log(value)

        return Logger(log)

    @classmethod
    def for_stdout(cls) -> "Logger":
        return Logger(lambda name, value: print(f"{name}: {value}"))
