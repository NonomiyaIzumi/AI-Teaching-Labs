"""
Cac ham bai tap (Exercise 1-4) - Gradient Checking.
Chuyen the tu 02_Kiem_tra_gradient.ipynb, giu nguyen cong thuc.

Luu y: backward_propagation_n() duoi day CO 2 BUG CO Y (dong bo goc trong Coursera):
- dW2 nhan thua "* 2"
- db1 dung "4./m" thay vi "1./m"
Day la muc dich su pham cua bai: chay gradient_check_n() se phat hien ra sai lech
(difference ~ 0.285, vuot nguong 2e-7) de minh hoa gradient checking hoat dong nhu the nao.
"""

import numpy as np

from utils import dictionary_to_vector, gradients_to_vector, relu, sigmoid, vector_to_dictionary


# Exercise 1
def forward_propagation(x, theta):
    """J(theta) = theta * x."""
    J = theta * x
    return J


# Exercise 2
def backward_propagation(x, theta):
    """dJ/dtheta = x."""
    dtheta = x
    return dtheta


# Exercise 3
def gradient_check(x, theta, epsilon=1e-7, print_msg=False):
    """Kiem tra dao ham so voi mo hinh 1D."""
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    J_plus = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    return difference


def forward_propagation_n(X, Y, parameters):
    """Cho san (khong phai bai tap): LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID + cost."""
    m = X.shape[1]
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    log_probs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1.0 / m * np.sum(log_probs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache


def backward_propagation_n(X, Y, cache):
    """Cho san (khong phai bai tap) - CO 2 BUG CO Y, xem docstring o dau file."""
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1.0 / m * np.dot(dZ3, A2.T)
    db3 = 1.0 / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1.0 / m * np.dot(dZ2, A1.T) * 2  # BUG co y: nhan thua "* 2"
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1.0 / m * np.dot(dZ1, X.T)
    db1 = 4.0 / m * np.sum(dZ1, axis=1, keepdims=True)  # BUG co y: phai la "1./m"

    gradients = {
        "dZ3": dZ3, "dW3": dW3, "db3": db3,
        "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
        "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1,
    }
    return gradients


# Exercise 4
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7, print_msg=False):
    """Kiem tra dao ham so voi mo hinh N chieu (nhieu tham so)."""
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i] = theta_plus[i] + epsilon
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i] = theta_minus[i] - epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if print_msg:
        if difference > 2e-7:
            print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
        else:
            print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    return difference
