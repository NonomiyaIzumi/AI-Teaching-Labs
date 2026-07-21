import numpy as np
import sklearn.model_selection


def load_dataset(csv_path="../../data/pima-indians-diabetes.csv", test_size=0.2, seed=1):
    """
    [Bonus] Bo du lieu Pima Indians Diabetes (768 benh nhan, 8 dac trung lam sang, du doan
    tieu duong) - dung de ap dung cac ham L-layer vua xay trong bai nay vao du lieu that.

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


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

