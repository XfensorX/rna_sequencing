import dataclasses
from typing import Iterator

import numpy as np
import numpy.typing as npt
from sklearn import metrics

from sklearn.exceptions import UndefinedMetricWarning


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
        Initializes the Metrics object by calculating various metrics using the input arrays.
        """
        self.accuracy = metrics.accuracy_score(y_actual, y_predicted > 0.5)
        self.precision = metrics.precision_score(
            y_actual, y_predicted, average="micro", zero_division=0
        )
        self.recall = metrics.recall_score(y_actual, y_predicted, average="micro")
        try:
            self.auc = metrics.roc_auc_score(
                y_actual, y_predicted, average="macro", multi_class="ovr"
            )
        except UndefinedMetricWarning:
            self.auc = 0.5

        self.f1_score = metrics.f1_score(y_actual, y_predicted, average="micro")

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """
        Allows iteration over the metric names and their respective values, yielding tuples
        of the form (name, value).
        """
        yield from dataclasses.asdict(self).items()
