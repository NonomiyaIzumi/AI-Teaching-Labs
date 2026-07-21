"""
Cac ham bai tap (Exercise 1-8) - Gradient Descent / Momentum / Adam / learning-rate decay,
cong model() (cho san, ban day du co ho tro decay) - chuyen the tu 04_Toi_uu_hoa.ipynb.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from utils import backward_propagation, compute_cost, forward_propagation, initialize_parameters


# Exercise 1
def update_parameters_with_gd(parameters, grads, learning_rate):
    """Gradient descent thuong: W -= lr*dW, b -= lr*db."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    return parameters


# Exercise 2
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """Xao tron va chia (X, Y) thanh cac mini-batch kich thuoc mini_batch_size."""
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1, m))

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size :]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches


# Exercise 3
def initialize_velocity(parameters):
    """Khoi tao v (van toc, dung cho Momentum) bang 0, cung shape voi W, b."""
    L = len(parameters) // 2
    v = {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v


# Exercise 4
def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """Cap nhat tham so bang Momentum."""
    L = len(parameters) // 2
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta * v["dW" + str(l)] + (1 - beta) * grads["dW" + str(l)]
        v["db" + str(l)] = beta * v["db" + str(l)] + (1 - beta) * grads["db" + str(l)]
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v["db" + str(l)]
    return parameters, v


# Exercise 5
def initialize_adam(parameters):
    """Khoi tao v va s (dung cho Adam) bang 0, cung shape voi W, b."""
    L = len(parameters) // 2
    v, s = {}, {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        s["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        s["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
    return v, s


# Exercise 6
def update_parameters_with_adam(
    parameters, grads, v, s, t, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8
):
    """Cap nhat tham so bang Adam (moment bac 1 + bac 2, co bias correction)."""
    L = len(parameters) // 2
    v_corrected, s_corrected = {}, {}
    for l in range(1, L + 1):
        v["dW" + str(l)] = beta1 * v["dW" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v["db" + str(l)] + (1 - beta1) * grads["db" + str(l)]

        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)

        s["dW" + str(l)] = beta2 * s["dW" + str(l)] + (1 - beta2) * (grads["dW" + str(l)] ** 2)
        s["db" + str(l)] = beta2 * s["db" + str(l)] + (1 - beta2) * (grads["db" + str(l)] ** 2)

        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * v_corrected["dW" + str(l)] / (
            np.sqrt(s_corrected["dW" + str(l)]) + epsilon
        )
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * v_corrected["db" + str(l)] / (
            np.sqrt(s_corrected["db" + str(l)]) + epsilon
        )

    return parameters, v, s, v_corrected, s_corrected


# Exercise 7
def update_lr(learning_rate0, epoch_num, decay_rate):
    """Exponential weight decay."""
    learning_rate = (1 / (1 + decay_rate * epoch_num)) * learning_rate0
    return learning_rate


# Exercise 8
def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """Exponential weight decay, chi cap nhat sau moi time_interval epoch."""
    learning_rate = (1 / (1 + decay_rate * np.floor(epoch_num / time_interval))) * learning_rate0
    return learning_rate


def model(
    X, Y, layers_dims, optimizer,
    learning_rate=0.0007, mini_batch_size=64, beta=0.9,
    beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=5000, print_cost=True,
    decay=None, decay_rate=1,
):
    """Mang 3 lop dung chung cho ca 3 optimizer (cho san, khong phai bai tap)."""
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]
    learning_rate0 = learning_rate

    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            a3, caches = forward_propagation(minibatch_X, parameters)
            cost_total += compute_cost(a3, minibatch_Y)
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s, _, _ = update_parameters_with_adam(
                    parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon
                )
        cost_avg = cost_total / m
        if decay:
            learning_rate = decay(learning_rate0, i, decay_rate)

        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
            if decay:
                print("learning rate after epoch %i: %f" % (i, learning_rate))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("epochs (per 100)")
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
