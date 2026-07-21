"""
Batch Normalization & Layer Normalization - chuyen the tu 05_Batch_Norm_Layer_Norm.ipynb.

Khac voi lab 01-04 (bai tap "dien vao cho trong"), notebook nay la code hoan chinh,
da duoc tac gia goc kiem chung bang gradient checking - khong co bug/stub can sinh vien tu viet.
"""

import numpy as np

from utils import relu, relu_backward, sigmoid


# Exercise 2.1
def batchnorm_forward(Z, gamma, beta, bn_param, mode):
    """
    Z: pre-activation, shape (n, m). gamma, beta: shape (n, 1).
    bn_param: dict luu running_mean/running_var/momentum/eps giua cac lan goi.
    mode: 'train' hoac 'test'.
    """
    eps = bn_param.get("eps", 1e-8)
    momentum = bn_param.get("momentum", 0.9)
    n, m = Z.shape
    running_mean = bn_param.setdefault("running_mean", np.zeros((n, 1)))
    running_var = bn_param.setdefault("running_var", np.zeros((n, 1)))

    if mode == "train":
        mu = np.mean(Z, axis=1, keepdims=True)
        var = np.var(Z, axis=1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + eps)
        Z_norm = (Z - mu) * std_inv
        out = gamma * Z_norm + beta

        cache = (Z, Z_norm, mu, var, std_inv, gamma, eps)
        bn_param["running_mean"] = momentum * running_mean + (1 - momentum) * mu
        bn_param["running_var"] = momentum * running_var + (1 - momentum) * var

    elif mode == "test":
        Z_norm = (Z - running_mean) / np.sqrt(running_var + eps)
        out = gamma * Z_norm + beta
        cache = None

    else:
        raise ValueError("mode phai la 'train' hoac 'test'")

    return out, cache


# Exercise 2.2
def batchnorm_backward(dZ_tilde, cache):
    """Chain rule qua Z -> mu,var -> Z_norm -> Z_tilde. Tra ve dZ, dgamma, dbeta."""
    Z, Z_norm, mu, var, std_inv, gamma, eps = cache
    m = Z.shape[1]
    x_mu = Z - mu

    dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
    dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True)

    dZ_norm = dZ_tilde * gamma
    dvar = np.sum(dZ_norm * x_mu, axis=1, keepdims=True) * -0.5 * std_inv**3
    dmu = np.sum(dZ_norm * -std_inv, axis=1, keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=1, keepdims=True)
    dZ = dZ_norm * std_inv + dvar * 2 * x_mu / m + dmu / m

    return dZ, dgamma, dbeta


def numerical_gradient(f, x, eps=1e-5):
    """Uoc luong dL/dx bang central difference, tung phan tu cua x."""
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    return np.max(np.abs(a - b) / (np.maximum(1e-8, np.abs(a) + np.abs(b))))


def layernorm_forward(Z, gamma, beta, eps=1e-8):
    """Giong batchnorm_forward nhung chuan hoa theo axis=0 (feature), khong phan biet train/test."""
    mu = np.mean(Z, axis=0, keepdims=True)
    var = np.var(Z, axis=0, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    Z_norm = (Z - mu) * std_inv
    out = gamma * Z_norm + beta

    cache = (Z, Z_norm, mu, var, std_inv, gamma, eps)
    return out, cache


def layernorm_backward(dZ_tilde, cache):
    """Cung cong thuc voi batchnorm_backward, doi axis=1 -> axis=0 va m -> n (so feature)."""
    Z, Z_norm, mu, var, std_inv, gamma, eps = cache
    n = Z.shape[0]
    x_mu = Z - mu

    dbeta = np.sum(dZ_tilde, axis=1, keepdims=True)
    dgamma = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True)

    dZ_norm = dZ_tilde * gamma
    dvar = np.sum(dZ_norm * x_mu, axis=0, keepdims=True) * -0.5 * std_inv**3
    dmu = np.sum(dZ_norm * -std_inv, axis=0, keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=0, keepdims=True)
    dZ = dZ_norm * std_inv + dvar * 2 * x_mu / n + dmu / n

    return dZ, dgamma, dbeta


def initialize_parameters(layers_dims, init_type="random_bad"):
    """
    init_type='random_bad': W *= 4 (co tinh, de tao vanishing/exploding gradient - xem lab 03).
    init_type='he': He initialization.
    Moi hidden layer them gamma=1, beta=0 (dung khi co normalization; vo hai neu norm_type='none').
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        scale = 4 if init_type == "random_bad" else np.sqrt(2.0 / layers_dims[l - 1])
        parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * scale
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))
        if l < L:
            parameters[f"gamma{l}"] = np.ones((layers_dims[l], 1))
            parameters[f"beta{l}"] = np.zeros((layers_dims[l], 1))
    return parameters


def forward_propagation(X, parameters, layers_dims, norm_type, bn_states, mode="train"):
    """norm_type: 'none' | 'batchnorm' | 'layernorm'. bn_states: {layer_index: bn_param}."""
    L = len(layers_dims) - 1
    A_prev = X
    caches = {}

    for l in range(1, L):
        W, b = parameters[f"W{l}"], parameters[f"b{l}"]
        Z = np.dot(W, A_prev) + b

        norm_cache = None
        Z_tilde = Z
        if norm_type == "batchnorm":
            gamma, beta = parameters[f"gamma{l}"], parameters[f"beta{l}"]
            Z_tilde, norm_cache = batchnorm_forward(Z, gamma, beta, bn_states[l], mode)
        elif norm_type == "layernorm":
            gamma, beta = parameters[f"gamma{l}"], parameters[f"beta{l}"]
            Z_tilde, norm_cache = layernorm_forward(Z, gamma, beta)

        A = relu(Z_tilde)
        caches[l] = (A_prev, W, b, Z_tilde, norm_cache, A)
        A_prev = A

    WL, bL = parameters[f"W{L}"], parameters[f"b{L}"]
    ZL = np.dot(WL, A_prev) + bL
    AL = sigmoid(ZL)
    caches[L] = (A_prev, WL, bL, ZL, None, AL)

    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]
    logprobs = np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    return (1.0 / m) * np.nansum(logprobs)


def backward_propagation(X, Y, caches, layers_dims, norm_type):
    L = len(layers_dims) - 1
    grads = {}

    A_prev, WL, bL, ZL, _, AL = caches[L]
    m = X.shape[1]
    dZ = (1.0 / m) * (AL - Y)
    grads[f"dW{L}"] = np.dot(dZ, A_prev.T)
    grads[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(WL.T, dZ)

    for l in reversed(range(1, L)):
        A_prev, W, b, Z_tilde, norm_cache, A = caches[l]
        dZ_tilde = relu_backward(dA_prev, Z_tilde)

        if norm_type == "batchnorm":
            dZ, dgamma, dbeta = batchnorm_backward(dZ_tilde, norm_cache)
            grads[f"dgamma{l}"], grads[f"dbeta{l}"] = dgamma, dbeta
        elif norm_type == "layernorm":
            dZ, dgamma, dbeta = layernorm_backward(dZ_tilde, norm_cache)
            grads[f"dgamma{l}"], grads[f"dbeta{l}"] = dgamma, dbeta
        else:
            dZ = dZ_tilde

        grads[f"dW{l}"] = np.dot(dZ, A_prev.T)
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

    return grads


def update_parameters(parameters, grads, learning_rate, layers_dims, norm_type):
    L = len(layers_dims) - 1
    for l in range(1, L + 1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
        if norm_type in ("batchnorm", "layernorm") and l < L:
            parameters[f"gamma{l}"] -= learning_rate * grads[f"dgamma{l}"]
            parameters[f"beta{l}"] -= learning_rate * grads[f"dbeta{l}"]
    return parameters


def model(X, Y, layers_dims, norm_type="none", init_type="random_bad", learning_rate=0.1, num_iterations=3000, track_every=10):
    """Huan luyen + ghi lai ||dW|| moi track_every iteration (gradient flow tracking)."""
    parameters = initialize_parameters(layers_dims, init_type)
    L = len(layers_dims) - 1
    bn_states = {l: {} for l in range(1, L)}

    grad_norms = {f"dW{l}": [] for l in range(1, L + 1)}
    tracked_iters, costs = [], []

    for i in range(num_iterations):
        AL, caches = forward_propagation(X, parameters, layers_dims, norm_type, bn_states, mode="train")
        cost = compute_cost(AL, Y)
        grads = backward_propagation(X, Y, caches, layers_dims, norm_type)
        parameters = update_parameters(parameters, grads, learning_rate, layers_dims, norm_type)

        if i % track_every == 0:
            tracked_iters.append(i)
            costs.append(cost)
            for l in range(1, L + 1):
                grad_norms[f"dW{l}"].append(np.linalg.norm(grads[f"dW{l}"]))

    return parameters, bn_states, costs, grad_norms, tracked_iters


def predict_dec(X, parameters, layers_dims, norm_type, bn_states):
    """Du doan nhi phan (nguong 0.5) dung cho ve decision boundary, dung mode='test'."""
    AL, _ = forward_propagation(X, parameters, layers_dims, norm_type, bn_states, mode="test")
    return AL > 0.5
