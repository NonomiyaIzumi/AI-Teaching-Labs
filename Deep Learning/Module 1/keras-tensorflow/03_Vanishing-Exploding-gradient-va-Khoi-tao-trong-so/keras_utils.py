"""
Cac ham dung chung cho notebook Keras/TensorFlow (da cung cap san, khong phai bai tap).
load_dataset() dung chung dinh dang (n_x, m)/(1, m) voi cac bai NumPy from-scratch trong
Module 1; compile_and_train()/evaluate_classification() la phan "hau ky" (compile, fit,
danh gia) khong doi giua cac bai, nen dat san o day thay vi bat sinh vien lam lai moi bai.
"""

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import sklearn.model_selection


def load_dataset(csv_path="../../data/pima-indians-diabetes.csv", test_size=0.2, seed=1):
    """
    Bo du lieu Pima Indians Diabetes (768 benh nhan, 8 dac trung lam sang, du doan tieu duong)
    - cung mot bo du lieu va cach tien xu ly nhu cac bai NumPy from-scratch trong Module 1.

    Returns: train_X, train_Y (shape (8, m_train), (1, m_train)), test_X, test_Y (tuong tu voi m_test).
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


def compile_and_train(model, X_train, Y_train, optimizer, epochs=100, batch_size=None, verbose=0):
    """
    Bien mot Sequential model chua compile thanh model da train.
    X_train shape (n_x, m), Y_train shape (1, m) - can .T vi Keras quy uoc sample-first.
    """
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train.T, Y_train.T, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return history


def evaluate_classification(model, X, y, title):
    """
    Du doan bang model.predict(), in Accuracy/Precision/Recall/F1 va ve confusion matrix + ROC curve.
    """
    y_true = y.ravel().astype(int)
    y_prob = model.predict(X.T, verbose=0).ravel()
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
