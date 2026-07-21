"""
Chay tuan tu 10 exercise cua bai 01, tuong duong chay het notebook tu tren xuong.
In ket qua de doi chieu voi "Expected output" trong tai lieu huong dan.
"""

import numpy as np

from model import (
    L_model_backward,
    L_model_forward,
    compute_cost,
    initialize_parameters,
    initialize_parameters_deep,
    linear_activation_backward,
    linear_activation_forward,
    linear_backward,
    linear_forward,
    update_parameters,
)
from public_tests import (
    L_model_backward_test,
    L_model_forward_test,
    compute_cost_test,
    initialize_parameters_deep_test_1,
    initialize_parameters_deep_test_2,
    initialize_parameters_test_1,
    initialize_parameters_test_2,
    linear_activation_backward_test,
    linear_activation_forward_test,
    linear_backward_test,
    linear_forward_test,
    update_parameters_test,
)
from test_cases import (
    L_model_backward_test_case,
    L_model_forward_test_case_2hidden,
    compute_cost_test_case,
    linear_activation_backward_test_case,
    linear_activation_forward_test_case,
    linear_backward_test_case,
    linear_forward_test_case,
    update_parameters_test_case,
)


def main():
    np.random.seed(1)

    print("=== Exercise 1: initialize_parameters ===")
    parameters = initialize_parameters(3, 2, 1)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    initialize_parameters_test_1(initialize_parameters)

    parameters = initialize_parameters(4, 3, 2)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    initialize_parameters_test_2(initialize_parameters)

    print("\n=== Exercise 2: initialize_parameters_deep ===")
    parameters = initialize_parameters_deep([5, 4, 3])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    initialize_parameters_deep_test_1(initialize_parameters_deep)

    parameters = initialize_parameters_deep([4, 3, 2])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
    initialize_parameters_deep_test_2(initialize_parameters_deep)

    print("\n=== Exercise 3: linear_forward ===")
    t_A, t_W, t_b = linear_forward_test_case()
    t_Z, _ = linear_forward(t_A, t_W, t_b)
    print("Z = " + str(t_Z))
    linear_forward_test(linear_forward)

    print("\n=== Exercise 4: linear_activation_forward ===")
    t_A_prev, t_W, t_b = linear_activation_forward_test_case()
    t_A, _ = linear_activation_forward(t_A_prev, t_W, t_b, activation="sigmoid")
    print("With sigmoid: A = " + str(t_A))
    t_A, _ = linear_activation_forward(t_A_prev, t_W, t_b, activation="relu")
    print("With ReLU: A = " + str(t_A))
    linear_activation_forward_test(linear_activation_forward)

    print("\n=== Exercise 5: L_model_forward ===")
    t_X, t_parameters = L_model_forward_test_case_2hidden()
    t_AL, t_caches = L_model_forward(t_X, t_parameters)
    print("AL = " + str(t_AL))
    L_model_forward_test(L_model_forward)

    print("\n=== Exercise 6: compute_cost ===")
    t_Y, t_AL = compute_cost_test_case()
    t_cost = compute_cost(t_AL, t_Y)
    print("Cost: " + str(t_cost))
    compute_cost_test(compute_cost)

    print("\n=== Exercise 7: linear_backward ===")
    t_dZ, t_linear_cache = linear_backward_test_case()
    t_dA_prev, t_dW, t_db = linear_backward(t_dZ, t_linear_cache)
    print("dA_prev: " + str(t_dA_prev))
    print("dW: " + str(t_dW))
    print("db: " + str(t_db))
    linear_backward_test(linear_backward)

    print("\n=== Exercise 8: linear_activation_backward ===")
    t_dAL, t_linear_activation_cache = linear_activation_backward_test_case()
    t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation="sigmoid")
    print("With sigmoid: dA_prev = " + str(t_dA_prev))
    print("With sigmoid: dW = " + str(t_dW))
    print("With sigmoid: db = " + str(t_db))
    t_dA_prev, t_dW, t_db = linear_activation_backward(t_dAL, t_linear_activation_cache, activation="relu")
    print("With relu: dA_prev = " + str(t_dA_prev))
    print("With relu: dW = " + str(t_dW))
    print("With relu: db = " + str(t_db))
    linear_activation_backward_test(linear_activation_backward)

    print("\n=== Exercise 9: L_model_backward ===")
    t_AL, t_Y_assess, t_caches = L_model_backward_test_case()
    grads = L_model_backward(t_AL, t_Y_assess, t_caches)
    print("dA0 = " + str(grads["dA0"]))
    print("dA1 = " + str(grads["dA1"]))
    print("dW1 = " + str(grads["dW1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db1 = " + str(grads["db1"]))
    print("db2 = " + str(grads["db2"]))
    L_model_backward_test(L_model_backward)

    print("\n=== Exercise 10: update_parameters ===")
    t_parameters, grads = update_parameters_test_case()
    t_parameters = update_parameters(t_parameters, grads, 0.1)
    print("W1 = " + str(t_parameters["W1"]))
    print("b1 = " + str(t_parameters["b1"]))
    print("W2 = " + str(t_parameters["W2"]))
    print("b2 = " + str(t_parameters["b2"]))
    update_parameters_test(update_parameters)

    print("\nHoan tat: toan bo 10 exercise da chay va pass test.")


if __name__ == "__main__":
    main()
