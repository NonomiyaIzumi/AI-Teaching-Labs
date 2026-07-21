"""
Chay tuan tu 4 exercise cua bai 02, tuong duong chay het notebook tu tren xuong.
"""

import numpy as np

from model import (
    backward_propagation,
    backward_propagation_n,
    forward_propagation,
    forward_propagation_n,
    gradient_check,
    gradient_check_n,
)
from test_cases import gradient_check_n_test_case
from utils import load_dataset


def main():
    print("=== Exercise 1: forward_propagation ===")
    x, theta = 2, 4
    J = forward_propagation(x, theta)
    print("J = " + str(J))

    print("\n=== Exercise 2: backward_propagation ===")
    x, theta = 3, 4
    dtheta = backward_propagation(x, theta)
    print("dtheta = " + str(dtheta))

    print("\n=== Exercise 3: gradient_check (1D) ===")
    x, theta = 3, 4
    gradient_check(x, theta, print_msg=True)

    print("\n=== Exercise 4: gradient_check_n (N-D, phat hien bug co y) ===")
    X, Y, parameters = gradient_check_n_test_case()
    cost, cache = forward_propagation_n(X, Y, parameters)
    gradients = backward_propagation_n(X, Y, cache)
    difference = gradient_check_n(parameters, gradients, X, Y, 1e-7, True)

    expected_values = [0.2850931567761623, 1.1890913024229996e-07]
    assert not (type(difference) == np.ndarray), "You are not using np.linalg.norm for numerator or denominator"
    assert type(difference) == np.float64, "The value must be a 64 bit floating point scalar."
    assert np.any(np.isclose(difference, expected_values)), "Wrong value. It is not one of the expected values"

    print("\nHoan tat: gradient_check_n da phat hien dung bug trong backward_propagation_n (nhu thiet ke).")

    print("\n=== [Bonus] Gradient checking tren du lieu that (Pima Indians Diabetes) ===")
    train_X, train_Y, test_X, test_Y = load_dataset()

    np.random.seed(1)
    batch_idx = np.random.choice(train_X.shape[1], size=16, replace=False)
    X_real = train_X[:, batch_idx]
    Y_real = train_Y[:, batch_idx]
    print("X_real:", X_real.shape, " Y_real:", Y_real.shape)

    np.random.seed(3)
    layers_dims_real = [8, 5, 3, 1]
    parameters_real = {}
    for l in range(1, len(layers_dims_real)):
        parameters_real[f"W{l}"] = np.random.randn(layers_dims_real[l], layers_dims_real[l - 1]) * 0.1
        parameters_real[f"b{l}"] = np.zeros((layers_dims_real[l], 1))

    cost_real, cache_real = forward_propagation_n(X_real, Y_real, parameters_real)
    gradients_real = backward_propagation_n(X_real, Y_real, cache_real)
    gradient_check_n(parameters_real, gradients_real, X_real, Y_real, 1e-7, True)


if __name__ == "__main__":
    main()
