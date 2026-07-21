"""
Chay tuan tu bai 03: 3 exercise khoi tao tham so + train/so sanh 3 cach khoi tao +
phan bonus debug/visualize gradient flow. Tuong duong chay het notebook tu tren xuong.
"""

import matplotlib.pyplot as plt
import numpy as np

from model import (
    initialize_parameters_he,
    initialize_parameters_random,
    initialize_parameters_zeros,
    model,
    model_with_grad_tracking,
)
from utils import evaluate_classification, load_dataset, predict


def train_and_show(train_X, train_Y, test_X, test_Y, initialization, title):
    parameters = model(train_X, train_Y, initialization=initialization)
    print("On the train set:")
    predict(train_X, train_Y, parameters)
    print("On the test set:")
    predict(test_X, test_Y, parameters)
    evaluate_classification(test_X, test_Y, parameters, title + " (test set)")


def main():
    train_X, train_Y, test_X, test_Y = load_dataset()

    print("=== Exercise 1: initialize_parameters_zeros (sanity check) ===")
    parameters = initialize_parameters_zeros([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    print("\n--- Train voi initialization = 'zeros' ---")
    train_and_show(train_X, train_Y, test_X, test_Y, "zeros", "Model with Zeros initialization")

    print("\n=== Exercise 2: initialize_parameters_random (sanity check) ===")
    parameters = initialize_parameters_random([3, 2, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    print("\n--- Train voi initialization = 'random' ---")
    train_and_show(train_X, train_Y, test_X, test_Y, "random", "Model with large random initialization")

    print("\n=== Exercise 3: initialize_parameters_he (sanity check) ===")
    parameters = initialize_parameters_he([2, 4, 1])
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    print("\n--- Train voi initialization = 'he' ---")
    train_and_show(train_X, train_Y, test_X, test_Y, "he", "Model with He initialization")

    print("\n=== [Bonus] Debug & Visualize Gradient Flow ===")
    histories = {}
    for init in ["zeros", "random", "he"]:
        _, grad_norms, tracked_iters = model_with_grad_tracking(
            train_X, train_Y, num_iterations=5000, initialization=init, track_every=10
        )
        histories[init] = {"grad_norms": grad_norms, "iters": tracked_iters}
        print(f"Done tracking gradients for initialization = '{init}'")

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colors = {"dW1": "tab:blue", "dW2": "tab:orange", "dW3": "tab:green"}
    for ax, init in zip(axes, ["zeros", "random", "he"]):
        h = histories[init]
        for layer_key, color in colors.items():
            ax.plot(h["iters"], h["grad_norms"][layer_key], label=layer_key, color=color)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(f'Gradient flow - initialization = "{init}"')
        ax.set_xlabel("iteration")
        ax.set_ylabel("||dW||  (log scale)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    final_norms = {
        init: [np.mean(histories[init]["grad_norms"][f"dW{l}"][-20:]) for l in [1, 2, 3]]
        for init in ["zeros", "random", "he"]
    }
    x = np.arange(3)
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, init in enumerate(["zeros", "random", "he"]):
        ax.bar(x + (i - 1) * width, final_norms[init], width, label=init)
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Layer 1", "Layer 2", "Layer 3"])
    ax.set_ylabel("Mean ||dW|| cuoi training (log scale)")
    ax.set_title("Gradient magnitude theo layer & cach khoi tao")
    ax.legend(title="Initialization")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nHoan tat bai 03.")


if __name__ == "__main__":
    main()
