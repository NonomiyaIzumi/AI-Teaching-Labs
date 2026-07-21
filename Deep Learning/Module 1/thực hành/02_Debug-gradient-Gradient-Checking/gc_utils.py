import numpy as np
import sklearn.model_selection


def load_dataset(csv_path="../../data/pima-indians-diabetes.csv", test_size=0.2, seed=1):
    """
    [Bonus] Bo du lieu Pima Indians Diabetes (768 benh nhan, 8 dac trung lam sang, du doan
    tieu duong) - dung de kiem tra gradient_check_n tren du lieu that thay vi chi tren
    tham so gia lap co dinh trong gradient_check_n_test_case().

    5 cot Glucose/BloodPressure/SkinThickness/Insulin/BMI khong the bang 0 ve mat sinh ly -
    gia tri 0 o day thuc chat la du lieu bi thieu. Dung median imputation (tinh tren tap train,
    ap dung lai cho test de tranh data leakage), roi chuan hoa (fit tren train).
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


def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)
    
    return s

# Duoc dictionary_to_vector() ghi lai moi lan goi, de vector_to_dictionary() biet cach
# khoi phuc dung shape ma khong can hard-code rieng cho kien truc W1:(5,4),b1:(5,1),...
# cua bai tap goc - nho vay 2 ham nay dung duoc voi BAT KY kien truc mang nao (vd bonus
# ap dung gradient checking cho mang train tren du lieu that o cuoi bai), khong doi API.
_last_shapes = {}

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    global _last_shapes
    keys = []
    count = 0
    _last_shapes = {}
    for key in parameters.keys():
        _last_shapes[key] = parameters[key].shape

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    Dung shape ghi lai boi lan goi dictionary_to_vector() gan nhat - luon goi dictionary_to_vector()
    truoc khi dung ham nay (dung nhu gradient_check_n() da lam).
    """
    parameters = {}
    idx = 0
    for key, shape in _last_shapes.items():
        size = int(np.prod(shape))
        parameters[key] = theta[idx: idx + size].reshape(shape)
        idx += size

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta