"""
Cac ham bai tap (Exercise 1-3) + model() (cho san) + model_with_grad_tracking() (bonus).
Chuyen the tu 03_Khoi_tao_trong_so.ipynb, giu nguyen cong thuc va random seed.
"""

import matplotlib.pyplot as plt
import numpy as np

from utils import backward_propagation, compute_loss, forward_propagation, update_parameters


# Exercise 1
def initialize_parameters_zeros(layers_dims):
    """Khoi tao toan bo W, b bang 0 (minh hoa vi sao fail symmetry breaking)."""
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# Exercise 2
def initialize_parameters_random(layers_dims):
    """Khoi tao W ngau nhien voi gia tri lon (*10), b bang 0."""
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


# Exercise 3
def initialize_parameters_he(layers_dims):
    """He initialization: W * sqrt(2 / n[l-1]), b bang 0."""
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(
            2.0 / layers_dims[l - 1]
        )
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    return parameters


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    """Mang 3 lop LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID (cho san, khong phai bai tap)."""
    costs = []
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (per hundreds)")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def model_with_grad_tracking(X, Y, learning_rate=0.01, num_iterations=5000, initialization="he", track_every=10):
    """
    [Bonus] Giong model() nhung ghi lai norm cua dW1, dW2, dW3 moi track_every iteration
    de ve gradient flow (phat hien vanishing/exploding gradient).
    """
    grad_norms = {"dW1": [], "dW2": [], "dW3": []}
    tracked_iters = []
    layers_dims = [X.shape[0], 10, 5, 1]

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):
        a3, cache = forward_propagation(X, parameters)
        compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % track_every == 0:
            tracked_iters.append(i)
            for l in [1, 2, 3]:
                grad_norms["dW" + str(l)].append(np.linalg.norm(grads["dW" + str(l)]))

    return parameters, grad_norms, tracked_iters
