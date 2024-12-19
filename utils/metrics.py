import dataclasses
from typing import Iterator

import numpy as np
import numpy.typing as npt
from sklearn import metrics


@dataclasses.dataclass
class Metrics:
    """
    A class to compute and store various performance metrics for model evaluation.

    Attributes
    ----------
    accuracy : float
        The proportion of correctly classified instances (both true positives and true negatives)
        among the total instances.
    precision : float
        The proportion of true positive instances among instances classified as positive.
    recall : float
        The proportion of true positive instances among the actual positive instances.
    auc : float
        The Area Under the Receiver Operating Characteristic Curve (ROC AUC), which measures
        the ability of the model to distinguish between classes.

    Methods
    -------
    __init__(y_predicted: npt.NDArray[np.float32], y_actual: npt.NDArray[np.float32])
        Initializes the Metrics object by calculating various metrics using the input arrays.
    __iter__() -> Iterator[Tuple[str, float]]:
        Allows iteration over the metric names and their respective values, yielding tuples
        of the form (name, value).
    """

    accuracy: float
    precision: float
    recall: float
    auc: float

    def __init__(
        self, y_predicted: npt.NDArray[np.float32], y_actual: npt.NDArray[np.float32]
    ):
        """
        Calculates and initializes various prediction metrics based on provided predictions and actual values.

        Parameters
        ----------
        y_predicted : npt.NDArray[np.float32]
            The predicted values, typically the output from a classifier.
        y_actual : npt.NDArray[np.float32]
            The actual ground truth values corresponding to the predictions.
        """
        self.accuracy = metrics.accuracy_score(y_actual, y_predicted)
        self.precision = metrics.precision_score(
            y_actual, y_predicted, average="micro", zero_division=0
        )
        self.recall = metrics.recall_score(y_actual, y_predicted, average="micro")
        self.auc = metrics.roc_auc_score(
            y_actual, y_predicted, average="macro", multi_class="ovr"
        )

    def __iter__(self) -> Iterator[tuple[str, float]]:
        """
        Returns an iterator over the metric names and their values.

        Yields
        ------
        Iterator[Tuple[str, float]]
            An iterator that yields tuples containing metric names and their values.
        """
        yield from dataclasses.asdict(self).items()
