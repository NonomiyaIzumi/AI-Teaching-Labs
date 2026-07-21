"""
Cac ham duoc cung cap san (khong phai bai tap) - port tu bn_ln_utils.py trong ban notebook.
"""

import numpy as np
import sklearn.datasets


def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """Compute the ReLU of x."""
    return np.maximum(0, x)


def relu_backward(dA, Z):
    """Backward pass cho ReLU: dZ = dA voi cac phan tu co Z <= 0 bi zero-out."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def load_dataset():
    """Bo du lieu 'circles' 2 lop (giong notebook 03)."""
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y
