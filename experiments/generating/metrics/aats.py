""" Script for Nearest Neighbor Adversarial Accuracy (AA)

Reference:
-----
Andrew Yale, Saloni Dash, Ritik Dutta, Isabelle Guyon, Adrien Pavao, Kristin P. Bennett,
Generation and evaluation of privacy preserving synthetic health data, Neurocomputing, Volume 416, 2020, Pages 244-255, ISSN 0925-2312,
https://doi.org/10.1016/j.neucom.2019.12.136. (https://www.sciencedirect.com/science/article/pii/S0925231220305117)

Original code:
-----
https://github.com/yknot/ESANN2019.git

"""

# Imports
from itertools import product
import concurrent.futures
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm import tqdm

# SEED
np.random.seed(42)

# AA class object


class NearestNeighborMetrics():
    """Calculate nearest neighbors and metrics"""

    def __init__(self, real_data, synths):
        self.data = {'train': real_data}

        # add all synthetics
        # for i, s in enumerate(synths):
        #self.data[f'synth_{i}'] = s.reshape(1,-1)
        #self.synth_keys = [f'synth_{i}' for i in range(len(synths))]
        self.data[f'synth_0'] = synths
        self.synth_keys = [f'synth_0']

        # pre allocate distances
        self.dists = {}

    def nearest_neighbors(self, t, s):
        """Find nearest neighbors d_ts and d_ss"""
        # fit to S
        nn_s = NearestNeighbors(n_neighbors=1).fit(self.data[s])
        if t == s:
            # find distances from s to s
            d = nn_s.kneighbors()[0]
        else:
            # find distances from t to s
            d = nn_s.kneighbors(self.data[t])[0]
        return t, s, d

    def compute_nn(self):
        """run all the nearest neighbors calculations"""
        tasks = product(self.data.keys(), repeat=2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [
                executor.submit(self.nearest_neighbors, t, s)
                for (t, s) in tasks
            ]
            # wait for each job to finish
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures)):
                t, s, d = future.result()
                self.dists[(t, s)] = d

    def divergence(self, t, s):
        """calculate the NN divergence"""
        left = np.mean(np.log(self.dists[(t, s)] / self.dists[(t, t)]))
        right = np.mean(np.log(self.dists[(s, t)] / self.dists[(s, s)]))
        return 0.5 * (left + right)

    def discrepancy_score(self, t, s):
        """calculate the NN discrepancy score"""
        left = np.mean(self.dists[(t, s)])
        right = np.mean(self.dists[(s, t)])
        return 0.5 * (left + right)

    def adversarial_accuracy(self, t, s):
        """calculate the NN adversarial accuracy"""
        left = np.mean(self.dists[(t, s)] > self.dists[(t, t)])
        right = np.mean(self.dists[(s, t)] > self.dists[(s, s)])
        return 0.5 * (left + right)

    def compute_discrepancy(self):
        """compute the standard discrepancy scores"""
        j_rr = self.discrepancy_score('train', 'train')
        j_ra = []
        j_aa = []
        # for all of the synthetic datasets
        for k in self.synth_keys:
            j_ra.append(self.discrepancy_score('train', k))
            # comparison to other synthetics
            for k_2 in self.synth_keys:
                if k != k_2:
                    j_aa.append(self.discrepancy_score(k, k_2))

        # average accross synthetics
        j_ra = np.mean(np.array(j_ra))
        j_aa = np.mean(np.array(j_aa))
        return j_rr, j_ra, j_aa

    def compute_divergence(self):
        """compute the standard divergence scores"""
        d_tr_a = []
        for k in self.synth_keys:
            d_tr_a.append(self.divergence('train', k))

        training = np.mean(np.array(d_tr_a))
        return training

    def compute_adversarial_accuracy(self):
        """compute the standarad adversarial accuracy scores"""
        a_tr_a = []
        for k in self.synth_keys:
            a_tr_a.append(self.adversarial_accuracy('train', k))

        a_tr = np.mean(np.array(a_tr_a))
        return a_tr

# Compute metric


def compute_AAts(real_data: np.array = None, fake_data: np.array = None):
    """ Compute similarity scores based on nearest neighbors distances.
    ----
    Parameters:
        real_data (np.array): array of real data
        fake_data (np.array): array of synthetic data
    Returns:
        discrepancy score, divergence score, adversarial accuracy
    """
    nnm = NearestNeighborMetrics(real_data, fake_data)
    nnm.compute_nn()

    # run discrepancy score, divergence, adversarial accuracy
    discrepancy = nnm.compute_discrepancy()
    #divergence = nnm.compute_divergence()
    adversarial = nnm.compute_adversarial_accuracy()

    return discrepancy[0], discrepancy[1], adversarial
