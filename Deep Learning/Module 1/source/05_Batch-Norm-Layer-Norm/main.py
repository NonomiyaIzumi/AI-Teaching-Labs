"""
Chay tuan tu bai 05: gradient check cho BatchNorm/LayerNorm, roi train + so sanh
3 phuong an chuan hoa (none/batchnorm/layernorm) tren mang 4 hidden layer voi khoi tao kem.
Tuong duong chay het notebook tu tren xuong.
"""

import matplotlib.pyplot as plt
import numpy as np

from model import (
    batchnorm_backward,
    batchnorm_forward,
    layernorm_backward,
    layernorm_forward,
    model,
    numerical_gradient,
    predict_dec,
    relative_error,
)
from utils import load_dataset


def check_batchnorm_gradient():
    print("=== Gradient check: BatchNorm ===")
    np.random.seed(0)
    n, m = 4, 5
    Z = np.random.randn(n, m) * 5
    gamma = np.random.randn(n, 1)
    beta = np.random.randn(n, 1)
    dout = np.random.randn(n, m)

    out, cache = batchnorm_forward(Z, gamma, beta, {}, mode="train")
    dZ, dgamma, dbeta = batchnorm_backward(dout, cache)

    def loss():
        o, _ = batchnorm_forward(Z, gamma, beta, {}, mode="train")
        return np.sum(o * dout)

    dZ_num = numerical_gradient(loss, Z)
    dgamma_num = numerical_gradient(loss, gamma)
    dbeta_num = numerical_gradient(loss, beta)

    print("BatchNorm gradient check:")
    print("  dZ     relative error =", relative_error(dZ, dZ_num))
    print("  dgamma relative error =", relative_error(dgamma, dgamma_num))
    print("  dbeta  relative error =", relative_error(dbeta, dbeta_num))
    assert relative_error(dZ, dZ_num) < 1e-6, "batchnorm_backward: dZ sai!"
    assert relative_error(dgamma, dgamma_num) < 1e-6, "batchnorm_backward: dgamma sai!"
    assert relative_error(dbeta, dbeta_num) < 1e-6, "batchnorm_backward: dbeta sai!"
    print("Tat ca deu dung (sai so < 1e-6).")


def check_layernorm_gradient():
    print("\n=== Gradient check: LayerNorm ===")
    np.random.seed(1)
    n, m = 4, 5
    Z = np.random.randn(n, m) * 5
    gamma = np.random.randn(n, 1)
    beta = np.random.randn(n, 1)
    dout = np.random.randn(n, m)

    out, cache = layernorm_forward(Z, gamma, beta)
    dZ, dgamma, dbeta = layernorm_backward(dout, cache)

    def loss():
        o, _ = layernorm_forward(Z, gamma, beta)
        return np.sum(o * dout)

    dZ_num = numerical_gradient(loss, Z)
    dgamma_num = numerical_gradient(loss, gamma)
    dbeta_num = numerical_gradient(loss, beta)

    print("LayerNorm gradient check:")
    print("  dZ     relative error =", relative_error(dZ, dZ_num))
    print("  dgamma relative error =", relative_error(dgamma, dgamma_num))
    print("  dbeta  relative error =", relative_error(dbeta, dbeta_num))
    assert relative_error(dZ, dZ_num) < 1e-6, "layernorm_backward: dZ sai!"
    assert relative_error(dgamma, dgamma_num) < 1e-6, "layernorm_backward: dgamma sai!"
    assert relative_error(dbeta, dbeta_num) < 1e-6, "layernorm_backward: dbeta sai!"
    print("Tat ca deu dung (sai so < 1e-6).")


def main():
    check_batchnorm_gradient()
    check_layernorm_gradient()

    print("\n=== Debug thuc te: mang sau (4 hidden layer) voi khoi tao kem ===")
    train_X, train_Y = load_dataset()
    layers_dims = [2, 20, 20, 20, 20, 1]

    histories = {}
    for norm_type in ["none", "batchnorm", "layernorm"]:
        parameters, bn_states, costs, grad_norms, tracked_iters = model(
            train_X, train_Y, layers_dims, norm_type=norm_type, init_type="random_bad",
            learning_rate=0.1, num_iterations=3000, track_every=10,
        )
        histories[norm_type] = {
            "parameters": parameters, "bn_states": bn_states,
            "costs": costs, "grad_norms": grad_norms, "iters": tracked_iters,
        }
        print(f"norm_type = {norm_type:10s} -> final cost = {costs[-1]:.4f}")

    print("\n--- Bieu do gradient flow: none vs batchnorm vs layernorm ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 1, 4))
    for ax, norm_type in zip(axes, ["none", "batchnorm", "layernorm"]):
        h = histories[norm_type]
        for l in range(1, 5):
            ax.plot(h["iters"], h["grad_norms"][f"dW{l}"], label=f"dW{l}", color=colors[l - 1])
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(f'Gradient flow - norm_type = "{norm_type}"')
        ax.set_xlabel("iteration")
        ax.set_ylabel("||dW|| (symlog)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("--- Duong cost qua qua trinh train (overlay 3 norm_type) ---")
    plt.figure(figsize=(7, 4.5))
    for norm_type in ["none", "batchnorm", "layernorm"]:
        plt.plot(histories[norm_type]["iters"], histories[norm_type]["costs"], label=norm_type)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("Cost qua qua trinh train")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("--- Decision boundary: none vs batchnorm vs layernorm ---")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x_min, x_max = train_X[0, :].min() - 1, train_X[0, :].max() + 1
    y_min, y_max = train_X[1, :].min() - 1, train_X[1, :].max() + 1
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].T

    for ax, norm_type in zip(axes, ["none", "batchnorm", "layernorm"]):
        h_ = histories[norm_type]
        Z = predict_dec(grid, h_["parameters"], layers_dims, norm_type, h_["bn_states"])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
        ax.scatter(train_X[0, :], train_X[1, :], c=train_Y, cmap=plt.cm.Spectral, edgecolors="k", s=15)
        ax.set_title(f'norm_type = "{norm_type}"  (cost={h_["costs"][-1]:.3f})')
    plt.tight_layout()
    plt.show()

    print("\nHoan tat bai 05.")


if __name__ == "__main__":
    main()
