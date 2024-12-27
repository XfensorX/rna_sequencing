import dataclasses
from typing import Iterator

import numpy as np
import numpy.typing as npt
from sklearn import metrics


@dataclasses.dataclass
class Metrics:
    """
    A class to compute and store various performance metrics for model evaluation.
    """

    accuracy: float
    precision: float
    recall: float
    auc: float
    f1_score: float

    def __init__(
        self, y_predicted: npt.NDArray[np.float32], y_actual: npt.NDArray[np.float32]
    ):
        """
        Initializes the Metrics object by calculating various metrics using the input arrays. This function uses macro averaging for precision, recall, and f1 score.
        """
        self.accuracy = self.calculate_accuracy(y_actual, y_predicted)
        self.precision = self.calculate_precision(y_actual, y_predicted)
        self.recall = self.calculate_recall(y_actual, y_predicted)
        self.auc = self.calculate_auc(y_actual, y_predicted)
        self.f1_score = self.calculate_f1_score(y_actual, y_predicted)

    @staticmethod
    def calculate_accuracy(
        y_actual: npt.NDArray[np.float32], y_predicted: npt.NDArray[np.float32]
    ) -> float:
        """
        Calculates the accuracy score.
        """
        return metrics.accuracy_score(y_actual, y_predicted)

    @staticmethod
    def calculate_precision(
        y_actual: npt.NDArray[np.float32], y_predicted: npt.NDArray[np.float32]
    ) -> float:
        """
        Calculates the precision score.
        """
        return metrics.precision_score(
            y_actual, y_predicted, average="macro", zero_division=0
        )

    @staticmethod
    def calculate_recall(
        y_actual: npt.NDArray[np.float32], y_predicted: npt.NDArray[np.float32]
    ) -> float:
        """
        Calculates the recall score.
        """
        return metrics.recall_score(y_actual, y_predicted, average="macro")

    @staticmethod
    def calculate_auc(
        y_actual: npt.NDArray[np.float32], y_predicted: npt.NDArray[np.float32]
    ) -> float:
        """
        Calculates the AUC score.
        """
        return metrics.roc_auc_score(
            y_actual, y_predicted, average="macro", multi_class="ovo"
        )

    @staticmethod
    def calculate_f1_score(
        y_actual: npt.NDArray[np.float32], y_predicted: npt.NDArray[np.float32]
    ) -> float:
        """
        Calculates the F1 score.
        """
        return metrics.f1_score(y_actual, y_predicted, average="macro")

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """
        Allows iteration over the metric names and their respective values, yielding tuples
        of the form (name, value).
        """
        yield from dataclasses.asdict(self).items()

    def as_dict(self) -> dict[str, float]:
        """
        Returns the metrics as a dictionary.
        """
        return dataclasses.asdict(self)
