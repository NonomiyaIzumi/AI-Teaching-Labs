"""
Cac ham duoc cung cap san (khong phai bai tap) - port tu init_utils.py trong ban notebook.
Bo load_cat_dataset() vi notebook nay khong dung toi va thieu file du lieu .h5 di kem.
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """Compute the relu of x."""
    return np.maximum(0, x)


def forward_propagation(X, parameters):
    """LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache


def backward_propagation(X, Y, cache):
    """Backprop cho kien truc co dinh o forward_propagation()."""
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1.0 / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dz3": dz3, "dW3": dW3, "db3": db3,
        "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
        "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1,
    }
    return gradients


def update_parameters(parameters, grads, learning_rate):
    """Gradient descent update."""
    L = len(parameters) // 2
    for k in range(L):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - learning_rate * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - learning_rate * grads["db" + str(k + 1)]
    return parameters


def compute_loss(a3, Y):
    """Cross-entropy loss."""
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    loss = 1.0 / m * np.nansum(logprobs)
    return loss


def predict(X, y, parameters):
    """Du doan tren tap X, in Accuracy va tra ve nhan du doan."""
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, _ = forward_propagation(X, parameters)

    for i in range(0, a3.shape[1]):
        p[0, i] = 1 if a3[0, i] > 0.5 else 0

    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))
    return p


def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel("x2")
    plt.xlabel("x1")
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


def predict_dec(parameters, X):
    """Du doan nhi phan (nguong 0.5) dung cho ve decision boundary."""
    a3, _ = forward_propagation(X, parameters)
    predictions = a3 > 0.5
    return predictions


def load_dataset():
    """Bo du lieu 'circles' 2 lop (xanh/do)."""
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=0.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=0.05)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y
