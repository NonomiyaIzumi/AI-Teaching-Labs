"""
Cac ham duoc cung cap san (khong phai bai tap) - port tu opt_utils_v1a.py trong ban notebook.
Bo load_params_and_grads() va load_2D_dataset() vi notebook nay khong dung toi.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics


def sigmoid(x):
    """Compute the sigmoid of x."""
    return 1 / (1 + np.exp(-x))


def relu(x):
    """Compute the relu of x."""
    return np.maximum(0, x)


def initialize_parameters(layer_dims):
    """He-style init cho mang 3 lop dung trong cac vi du minh hoa."""
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def forward_propagation(X, parameters):
    """LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID."""
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    z3 = np.dot(W3, a2) + b3
    a3 = sigmoid(z3)

    cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
    return a3, cache


def backward_propagation(X, Y, cache):
    """Backprop cho kien truc co dinh o forward_propagation()."""
    m = X.shape[1]
    (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache

    dz3 = 1.0 / m * (a3 - Y)
    dW3 = np.dot(dz3, a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)

    da2 = np.dot(W3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(W2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    gradients = {
        "dz3": dz3, "dW3": dW3, "db3": db3,
        "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
        "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1,
    }
    return gradients


def compute_cost(a3, Y):
    """Tong cross-entropy tren 1 mini-batch (chua chia cho m - goi tich luy roi chia sau)."""
    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost_total = np.sum(logprobs)
    return cost_total


def predict(X, y, parameters):
    """Du doan tren tap X, in Accuracy va tra ve nhan du doan."""
    m = X.shape[1]
    p = np.zeros((1, m), dtype=int)

    a3, _ = forward_propagation(X, parameters)

    for i in range(0, a3.shape[1]):
        p[0, i] = 1 if a3[0, i] > 0.5 else 0

    print("Accuracy: " + str(np.mean((p[0, :] == y[0, :]))))
    return p


def load_dataset(csv_path="../../data/pima-indians-diabetes.csv", test_size=0.2, seed=1):
    """
    Bo du lieu Pima Indians Diabetes (768 benh nhan, 8 dac trung lam sang, du doan tieu duong).

    5 cot Glucose/BloodPressure/SkinThickness/Insulin/BMI khong the bang 0 ve mat sinh ly -
    gia tri 0 o day thuc chat la du lieu bi thieu. Dung median imputation (tinh tren tap train,
    ap dung lai cho test de tranh data leakage), roi chuan hoa (fit tren train).

    Returns: train_X, train_Y (shape (8, m_train), (1, m_train)), test_X, test_Y (tuong tu voi m_test)
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


def evaluate_classification(X, y, parameters, title):
    """
    Du doan tren (X, y) bang mo hinh da train, in Accuracy/Precision/Recall/F1 va ve
    confusion matrix + ROC curve. Thay cho decision boundary 2D (khong con ve duoc
    truc tiep voi du lieu nhieu hon 2 chieu nhu Pima Diabetes).
    """
    a3, _ = forward_propagation(X, parameters)
    y_true = y.ravel().astype(int)
    y_prob = a3.ravel()
    y_pred = (y_prob > 0.5).astype(int)

    acc = (y_pred == y_true).mean()
    prec = sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)
    rec = sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)

    print(title)
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}  (ty le phat hien dung ca mac benh - quan trong trong bai toan y te)")
    print(f"  F1-score:  {f1:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["No Diabetes", "Diabetes"])
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["No Diabetes", "Diabetes"])
    axes[0].set_xlabel("Du doan")
    axes[0].set_ylabel("Thuc te")
    axes[0].set_title("Confusion Matrix\n" + title)

    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_prob)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    axes[1].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve\n" + title)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": roc_auc}
