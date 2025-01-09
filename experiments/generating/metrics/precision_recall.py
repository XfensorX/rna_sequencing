""" Script for k-NN Precision and Recall.

Reference:
-----
Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., & Aila, T. (2019).
Improved Precision and Recall Metric for Assessing Generative Models. ArXiv, abs/1904.06991.

Original Code:
------
Adaptation of tensorflow code provided by NVIDIA CORPORATION: https://github.com/kynkaat/improved-precision-and-recall-metric.git)
"""

# Imports
import os
import numpy as np
import torch
import time as t
import random

# SEED
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


def batch_pairwise_distances(U: torch.tensor, V: torch.tensor):
    """Compute pairwise distances between two batches of feature vectors.
    ----
    Parameters:
        U (torch.tensor): first feature vector
        V (torch.tensor): second feature vector
    Returns:
        tensor of pairwise distances"""

    # Squared norms of each row in U and V.
    norm_u = torch.sum(torch.square(U), 1)
    norm_v = torch.sum(torch.square(V), 1)

    # norm_u as a column and norm_v as a row vectors.
    norm_u = torch.reshape(norm_u, (-1, 1))
    norm_v = torch.reshape(norm_v, (1, -1))

    # Pairwise squared Euclidean distances.
    D = torch.maximum(norm_u - 2 * torch.matmul(U, V.T) + norm_v, torch.zeros((1,)))

    return D


class ManifoldEstimator:
    """Estimates the manifold of given feature vectors."""

    def __init__(
        self,
        features,
        row_batch_size=25000,
        col_batch_size=50000,
        nhood_sizes=[3],
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """Estimate the manifold of given feature vectors.

        Args:
            features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                the given percentile.
            eps (float): Small number for numerical stability.
        """
        batch_size = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features

        # Estimate manifold of features by calculating distances to k-NN of
        # each sample.
        self.D = np.zeros([batch_size, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, batch_size], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, batch_size, row_batch_size):
            end1 = min(begin1 + row_batch_size, batch_size)
            row_batch = features[begin1:end1]

            for begin2 in range(0, batch_size, col_batch_size):
                end2 = min(begin2 + col_batch_size, batch_size)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0 : end1 - begin1, begin2:end2] = (
                    batch_pairwise_distances(row_batch, col_batch).cpu()
                )

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(
                distance_batch[0 : end1 - begin1, :], seq, axis=1
            )[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval = eval_features.shape[0]
        num_ref = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref], dtype=np.float32)
        batch_predictions = np.zeros([num_eval, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros(
            [
                num_eval,
            ],
            dtype=np.float32,
        )
        nearest_indices = np.zeros(
            [
                num_eval,
            ],
            dtype=np.int32,
        )

        for begin1 in range(0, num_eval, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0 : end1 - begin1, begin2:end2] = (
                    batch_pairwise_distances(feature_batch, ref_batch).cpu()
                )

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of
            # neighborhood size k.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(
                np.int32
            )

            max_realism_score[begin1:end1] = np.max(
                self.D[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(
                distance_batch[0 : end1 - begin1, :], axis=1
            )

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


def knn_precision_recall_features(
    ref_features,
    eval_features,
    nhood_sizes=[3],
    row_batch_size=10000,
    col_batch_size=50000,
    num_gpus=1,
):
    """Calculates k-NN precision and recall for two sets of feature vectors.

    Args:
        ref_features (np.array/tf.Tensor): Feature vectors of reference samples.
        eval_features (np.array/tf.Tensor): Feature vectors of generated samples.
        nhood_sizes (list): Number of neighbors used to estimate the manifold.
        row_batch_size (int): Row batch size to compute pairwise distances
            (parameter to trade-off between memory usage and performance).
        col_batch_size (int): Column batch size to compute pairwise distances.
        num_gpus (int): Number of GPUs used to evaluate precision and recall.
    Returns:
        State (dict): Dict that contains precision and recall calculated from
        ref_features and eval_features.
    """
    state = dict()
    num_data = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize ManifoldEstimators.
    ref_manifold = ManifoldEstimator(
        ref_features, row_batch_size, col_batch_size, nhood_sizes
    )
    eval_manifold = ManifoldEstimator(
        eval_features, row_batch_size, col_batch_size, nhood_sizes
    )

    # Evaluate precision and recall using k-nearest neighbors.
    # print('Evaluating k-NN precision and recall with %i samples...' % num_data)
    start = t.time()

    # Precision: How many points from eval_features are in ref_features
    # manifold.
    precision = ref_manifold.evaluate(eval_features)
    state["precision"] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state["recall"] = recall.mean(axis=0)

    # print('Evaluated k-NN precision and recall in: %gs' % (t.time() - start))

    return state


def get_precision_recall(
    real_data: torch.tensor, fake_data: torch.tensor, nb_nn: list = [50]
):
    """
    Compute precision and recall between datasets.
    ----
    Parameters:
        real_data (torch.tensor): First data set of comparison.
        fake_data (torch.tensor): Second dataset to use for comparison.
        nb_nn (list): Number of neighbors used to estimate the data manifold.
    Returns:
        tuple with precision and recall.
    """

    # Calculate k-NN precision and recall.
    state = knn_precision_recall_features(real_data, fake_data, nhood_sizes=nb_nn)

    precision = state["precision"][0]
    recall = state["recall"][0]

    return (precision, recall)


def get_realism_score(real_data: torch.tensor, fake_data: torch.tensor):
    """
    Compute realism score between datasets.
    ----
    Parameters:
        real_data (torch.tensor): First data set of comparison.
        fake_data (torch.tensor): Second dataset to use for comparison.
    Returns:
        Maximum realism score.
    """

    # Estimate manifold of real images.
    print("Estimating manifold of real data...")
    real_manifold = ManifoldEstimator(real_data, clamp_to_percentile=50)

    # Estimate quality of individual samples.
    _, realism_scores = real_manifold.evaluate(fake_data, return_realism=True)

    return realism_scores
