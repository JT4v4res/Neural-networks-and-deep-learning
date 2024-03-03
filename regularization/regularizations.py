import numpy as np


def l1_regularization(weights, derivative=False):
    if derivative:
        weights = [np.where(w < 0, -1, w) for w in weights]
        return np.array([np.where(w > 0, 1, w) for w in weights])
    return np.sum([np.abs(w) for w in weights])


def l2_regularization(weights, derivative=False):
    if derivative:
        return weights
    return 0.5 ** np.sum(weights ** 2)
