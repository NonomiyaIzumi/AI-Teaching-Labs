"""
Chay tuan tu bai 04: 8 exercise ve toi uu hoa (GD/Momentum/Adam/LR decay) +
7 lan train mo hinh 3 lop de so sanh cac optimizer. Tuong duong chay het notebook tu tren xuong.
"""

from model import (
    initialize_adam,
    initialize_velocity,
    model,
    random_mini_batches,
    schedule_lr_decay,
    update_lr,
    update_parameters_with_adam,
    update_parameters_with_gd,
    update_parameters_with_momentum,
)
from test_cases import (
    initialize_adam_test_case,
    initialize_velocity_test_case,
    random_mini_batches_test_case,
    update_parameters_with_adam_test_case,
    update_parameters_with_gd_test_case,
    update_parameters_with_momentum_test_case,
)
from utils import evaluate_classification, load_dataset, predict


def train_and_show(train_X, train_Y, test_X, test_Y, layers_dims, title, **model_kwargs):
    parameters = model(train_X, train_Y, layers_dims, **model_kwargs)
    print("On the train set:")
    predict(train_X, train_Y, parameters)
    print("On the test set:")
    predict(test_X, test_Y, parameters)
    metrics = evaluate_classification(test_X, test_Y, parameters, title + " (test set)")
    return parameters, metrics


def main():
    train_X, train_Y, test_X, test_Y = load_dataset()
    layers_dims = [train_X.shape[0], 5, 2, 1]

    print("=== Exercise 1: update_parameters_with_gd ===")
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    parameters = update_parameters_with_gd(parameters, grads, 0.01)
    print("W1 =\n" + str(parameters["W1"]))
    print("b1 =\n" + str(parameters["b1"]))
    print("W2 =\n" + str(parameters["W2"]))
    print("b2 =\n" + str(parameters["b2"]))

    print("\n=== Exercise 2: random_mini_batches ===")
    t_X, t_Y, mini_batch_size = random_mini_batches_test_case()
    mini_batches = random_mini_batches(t_X, t_Y, mini_batch_size)
    print("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
    print("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
    print("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
    print("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
    print("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
    print("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
    print("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

    print("\n=== Exercise 3: initialize_velocity ===")
    parameters = initialize_velocity_test_case()
    v = initialize_velocity(parameters)
    print('v["dW1"] =\n' + str(v["dW1"]))
    print('v["db1"] =\n' + str(v["db1"]))
    print('v["dW2"] =\n' + str(v["dW2"]))
    print('v["db2"] =\n' + str(v["db2"]))

    print("\n=== Exercise 4: update_parameters_with_momentum ===")
    parameters, grads, v = update_parameters_with_momentum_test_case()
    parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=0.01)
    print("W1 = \n" + str(parameters["W1"]))
    print("b1 = \n" + str(parameters["b1"]))
    print("W2 = \n" + str(parameters["W2"]))
    print("b2 = \n" + str(parameters["b2"]))
    print('v["dW1"] = \n' + str(v["dW1"]))
    print('v["db1"] = \n' + str(v["db1"]))
    print('v["dW2"] = \n' + str(v["dW2"]))
    print('v["db2"] = \n' + str(v["db2"]))

    print("\n=== Exercise 5: initialize_adam ===")
    parameters = initialize_adam_test_case()
    v, s = initialize_adam(parameters)
    print('v["dW1"] = \n' + str(v["dW1"]))
    print('v["db1"] = \n' + str(v["db1"]))
    print('v["dW2"] = \n' + str(v["dW2"]))
    print('v["db2"] = \n' + str(v["db2"]))
    print('s["dW1"] = \n' + str(s["dW1"]))
    print('s["db1"] = \n' + str(s["db1"]))
    print('s["dW2"] = \n' + str(s["dW2"]))
    print('s["db2"] = \n' + str(s["db2"]))

    print("\n=== Exercise 6: update_parameters_with_adam ===")
    parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon = update_parameters_with_adam_test_case()
    parameters, v, s, vc, sc = update_parameters_with_adam(
        parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon
    )
    print(f"W1 = \n{parameters['W1']}")
    print(f"W2 = \n{parameters['W2']}")
    print(f"b1 = \n{parameters['b1']}")
    print(f"b2 = \n{parameters['b2']}")

    print("\n--- Train 3-layer model: so sanh gd / momentum / adam (khong decay) ---")
    train_and_show(train_X, train_Y, test_X, test_Y, layers_dims, "Model with Gradient Descent optimization", optimizer="gd")
    train_and_show(
        train_X, train_Y, test_X, test_Y, layers_dims, "Model with Momentum optimization", optimizer="momentum", beta=0.9
    )
    train_and_show(train_X, train_Y, test_X, test_Y, layers_dims, "Model with Adam optimization", optimizer="adam")

    print("\n=== Exercise 7: update_lr ===")
    learning_rate = 0.5
    print("Original learning rate: ", learning_rate)
    epoch_num = 2
    decay_rate = 1
    learning_rate_2 = update_lr(learning_rate, epoch_num, decay_rate)
    print("Updated learning rate: ", learning_rate_2)

    print("\n--- Train voi optimizer='gd', decay=update_lr ---")
    train_and_show(
        train_X, train_Y, test_X, test_Y, layers_dims, "Model with Gradient Descent optimization",
        optimizer="gd", learning_rate=0.1, num_epochs=5000, decay=update_lr,
    )

    print("\n=== Exercise 8: schedule_lr_decay ===")
    learning_rate = 0.5
    print("Original learning rate: ", learning_rate)
    epoch_num_1, epoch_num_2 = 10, 100
    decay_rate = 0.3
    time_interval = 100
    learning_rate_1 = schedule_lr_decay(learning_rate, epoch_num_1, decay_rate, time_interval)
    learning_rate_2 = schedule_lr_decay(learning_rate, epoch_num_2, decay_rate, time_interval)
    print("Updated learning rate after {} epochs: ".format(epoch_num_1), learning_rate_1)
    print("Updated learning rate after {} epochs: ".format(epoch_num_2), learning_rate_2)

    print("\n--- Train voi schedule_lr_decay: gd / momentum / adam ---")
    train_and_show(
        train_X, train_Y, test_X, test_Y, layers_dims, "Model with Gradient Descent optimization",
        optimizer="gd", learning_rate=0.1, num_epochs=5000, decay=schedule_lr_decay,
    )
    train_and_show(
        train_X, train_Y, test_X, test_Y, layers_dims, "Model with Gradient Descent with momentum optimization",
        optimizer="momentum", learning_rate=0.1, num_epochs=5000, decay=schedule_lr_decay,
    )
    train_and_show(
        train_X, train_Y, test_X, test_Y, layers_dims, "Model with Adam optimization",
        optimizer="adam", learning_rate=0.01, num_epochs=5000, decay=schedule_lr_decay,
    )

    print("\nHoan tat bai 04.")


if __name__ == "__main__":
    main()
