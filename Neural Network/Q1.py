import numpy as np
import pandas as pd
np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def initialize_weights(number_of_features, number_of_nodes, weights_init):
    if weights_init == "random":
        return [
            np.random.randn(number_of_features + 1, number_of_nodes),
            np.random.randn(number_of_nodes + 1, number_of_nodes),
            np.random.randn(number_of_nodes + 1, 1)
        ]
    else:
        return [
            np.zeros((number_of_features + 1, number_of_nodes)),
            np.zeros((number_of_nodes + 1, number_of_nodes)),
            np.zeros((number_of_nodes + 1, 1))
        ]

def learning_rate_schedule(initial_lr, d, epoch):
    return initial_lr / (1 + (initial_lr / d) * epoch)

def forward_pass(x, weights):
    activations = [x]
    z = []
    for layer_weights in weights:
        input_x = np.hstack((activations[-1], np.ones(1)))
        z.append(np.dot(input_x, layer_weights))
        activations.append(sigmoid(z[-1]))
    return activations, z

def backward_pass(y, activations, z, weights):
    delta = [activations[-1] - y]
    for i in range(len(weights) - 1, 0, -1):
        delta.append(np.dot(delta[-1], weights[i][:-1, :].T) * sigmoid_derivative(z[i - 1]))
    delta.reverse()
    return delta

def update_weights(weights, activations, delta, learning_rate):
    for i in range(len(weights)):
        input_x = np.hstack((activations[i], np.ones(1)))
        d_loss_w = np.dot(input_x[:, np.newaxis], delta[i][np.newaxis, :])
        weights[i] -= learning_rate * d_loss_w

def compute_loss(x, y, weights):
    loss = 0
    for i in range(len(x)):
        activations, _ = forward_pass(x[i], weights)
        loss += 0.5 * ((activations[-1] - y[i]) ** 2)
    return loss / len(x)

def train(x, y, weights, epochs, initial_lr, d):
    for epoch in range(epochs):
        lr = learning_rate_schedule(initial_lr, d, epoch)
        idx = np.random.permutation(len(x))
        train_x, train_y = x[idx], y[idx]
        for i in range(len(x)):
            activations, z = forward_pass(train_x[i], weights)
            delta = backward_pass(train_y[i], activations, z, weights)
            update_weights(weights, activations, delta, lr)
        loss = compute_loss(x, y, weights)
        print(f"Epoch: {epoch + 1}\t Training Loss: {loss}")

def predict(x, weights):
    for i in range(len(x)):
        activations, _ = forward_pass(x[i], weights)
    return 1 if activations[-1] >= 0.5 else 0

def evaluate(x, y, weights):
    predictions = [predict([x[i]], weights) for i in range(len(x))]
    return np.mean(np.array(predictions) != y)

if __name__ == "__main__":
    train_dataframe = pd.read_csv('/Users/mdrakibulhaque/Desktop/Neural Network/bank-note/train.csv', header=None)
    test_dataframe = pd.read_csv('/Users/mdrakibulhaque/Desktop/Neural Network/bank-note/test.csv', header=None)
    train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    train_x = train_dataframe.iloc[:, :-1].values
    train_y = train_dataframe.iloc[:, -1].values
    test_x = test_dataframe.iloc[:, :-1].values
    test_y = test_dataframe.iloc[:, -1].values

    lr = 0.0001
    d = 0.01
    T = 100
    number_of_nodes = [5, 10, 25, 50, 100]

    with open("results.txt", "w") as result_file:
        result_file.write("Number of Nodes\tRandom_Train_Error\tRandom_Test_Error\tZero_Train_Error\tZero_Test_Error\n")

        for number_of_node in number_of_nodes:
            print(f"Training with Number of Node={number_of_node}")

            weights_random = initialize_weights(train_x.shape[1], number_of_node, "random")
            train(train_x, train_y, weights_random, T, lr, d)
            random_training_error = round(evaluate(train_x, train_y, weights_random), 7)
            random_testing_error = round(evaluate(test_x, test_y, weights_random), 7)

            weights_zeros = initialize_weights(train_x.shape[1], number_of_node, "zeros")
            train(train_x, train_y, weights_zeros, T, lr, d)
            zero_training_error = round(evaluate(train_x, train_y, weights_zeros), 7)
            zero_testing_error = round(evaluate(test_x, test_y, weights_zeros), 7)

            result_file.write(f"{number_of_node}\t{random_training_error}\t{random_testing_error}\t{zero_training_error}\t{zero_testing_error}\n")

            print(f"Number of Nodes={number_of_node}\nRandom Init: Train Error={random_training_error}, Test Error={random_testing_error}\n"
                  f"Zero Init: Train Error={zero_training_error}, Test Error={zero_testing_error}\n")

