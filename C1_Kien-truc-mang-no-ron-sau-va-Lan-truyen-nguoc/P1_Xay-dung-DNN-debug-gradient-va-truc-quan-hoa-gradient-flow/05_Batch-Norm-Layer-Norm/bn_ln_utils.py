import numpy as np
import sklearn.datasets


def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """Compute the ReLU of x."""
    return np.maximum(0, x)


def relu_backward(dA, Z):
    """
    Backward pass for a ReLU unit.

    Arguments:
    dA -- gradient flowing back from the next layer, same shape as Z
    Z -- the input that was fed into relu() during the forward pass

    Returns:
    dZ -- gradient of the loss with respect to Z
    """
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def load_dataset():
    """Same 2-class 'circles' toy dataset used in the Initialization lab."""
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y
