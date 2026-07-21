"""
Cac ham bai tap (Exercise 1-10) - xay dung DNN L lop tu dau bang NumPy.
Chuyen the tu 01_Xay_dung_mo_hinh_DNN.ipynb, giu nguyen cong thuc va random seed.
"""

import copy

import numpy as np

from utils import relu, relu_backward, sigmoid, sigmoid_backward


# Exercise 1
def initialize_parameters(n_x, n_h, n_y):
    """Khoi tao tham so cho mang 2 lop."""
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# Exercise 2
def initialize_parameters_deep(layer_dims):
    """Khoi tao tham so cho mang L lop bat ky."""
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        assert parameters["W" + str(l)].shape == (layer_dims[l], layer_dims[l - 1])
        assert parameters["b" + str(l)].shape == (layer_dims[l], 1)
    return parameters


# Exercise 3
def linear_forward(A, W, b):
    """Phan LINEAR cua forward propagation: Z = W.A + b."""
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


# Exercise 4
def linear_activation_forward(A_prev, W, b, activation):
    """LINEAR -> ACTIVATION (sigmoid hoac relu)."""
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


# Exercise 5
def L_model_forward(X, parameters):
    """[LINEAR->RELU]*(L-1) -> LINEAR->SIGMOID."""
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation="relu"
        )
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid"
    )
    caches.append(cache)
    return AL, caches


# Exercise 6
def compute_cost(AL, Y):
    """Cross-entropy cost."""
    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost


# Exercise 7
def linear_backward(dZ, cache):
    """Phan LINEAR cua backward propagation cho 1 lop."""
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


# Exercise 8
def linear_activation_backward(dA, cache, activation):
    """LINEAR -> ACTIVATION backward."""
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


# Exercise 9
def L_model_backward(AL, Y, caches):
    """Backward propagation cho [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID."""
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L - 1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads["dA" + str(L - 1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l + 1)], current_cache, activation="relu"
        )
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


# Exercise 10
def update_parameters(params, grads, learning_rate):
    """Gradient descent step."""
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters
