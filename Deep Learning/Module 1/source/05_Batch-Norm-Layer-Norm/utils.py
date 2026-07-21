"""
Cac ham duoc cung cap san (khong phai bai tap) - port tu bn_ln_utils.py trong ban notebook.
"""

import numpy as np
import sklearn.datasets
import sklearn.model_selection


def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """Compute the ReLU of x."""
    return np.maximum(0, x)


def relu_backward(dA, Z):
    """Backward pass cho ReLU: dZ = dA voi cac phan tu co Z <= 0 bi zero-out."""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def load_dataset(csv_path="../../data/pima-indians-diabetes.csv", test_size=0.2, seed=1):
    """
    Bo du lieu Pima Indians Diabetes (768 benh nhan, 8 dac trung lam sang, du doan tieu duong).
    Giong lab 03/04: xu ly missing value an duoi dang so 0 (Glucose/BloodPressure/SkinThickness/
    Insulin/BMI) bang median imputation (fit tren train), roi chuan hoa (fit tren train).
    """
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    X = raw[:, :8]
    y = raw[:, 8]

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    zero_as_missing_cols = [1, 2, 3, 4, 5]  # Glucose, BloodPressure, SkinThickness, Insulin, BMI
    for c in zero_as_missing_cols:
        col = X_train[:, c]
        median = np.median(col[col != 0])
        X_train[X_train[:, c] == 0, c] = median
        X_test[X_test[:, c] == 0, c] = median

    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma

    train_X, test_X = X_train.T, X_test.T
    train_Y = y_train.reshape(1, -1)
    test_Y = y_test.reshape(1, -1)
    return train_X, train_Y, test_X, test_Y
