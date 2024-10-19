import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_data(train_filename, test_filename, column_names):
    train_data = pd.read_csv(train_filename, names=column_names).astype(float)
    test_data = pd.read_csv(test_filename, names=column_names).astype(float)
    y_train = train_data[column_names[-1]]
    y_test = test_data[column_names[-1]]
    X_train = train_data.drop(column_names[-1], axis=1)
    X_test = test_data.drop(column_names[-1], axis=1)

    return X_train, y_train, X_test, y_test

def initialize_weights(n_features):
    return np.zeros(n_features)  # +1 for the bias term

def compute_gradient(X, y, weights):
    predictions = np.dot(X, weights)
    errors = predictions - y
    gradient = np.dot(X.T ,errors) / len(X)
    return gradient

def batch_gradient_descent(X, y, learning_rate=0.01, iterations=1000, tolerance=1e-6):

    weights = initialize_weights(X.shape[1])
    cost_history = []

    for i in range(iterations):
        gradient = compute_gradient(X, y, weights)
        new_weights = weights - learning_rate * gradient
        cost = np.mean((np.dot(X, weights) - y) ** 2) / 2  
        cost_history.append(cost)

        # Check convergence
        if np.linalg.norm(new_weights - weights, 2) < tolerance:
            break

        weights = new_weights

    return weights, cost_history

def plot_cost(cost_history,lr):
    plt.plot(cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(f'Cost Function During Training with_{lr}')
    plt.savefig("Gradient_Loss.png")
    plt.show()

if __name__=="__main__":
    training_file="Concrete/train.csv"
    testing_file="Concrete/test.csv"
    column_names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'slump']
    X_train, y_train, X_test, y_test = get_data(training_file, testing_file, column_names)

    # Train model
    lr=0.01

    weights, cost_history = batch_gradient_descent(X_train, y_train, learning_rate=lr, iterations=8000)
    print("Final Weights for training are:")
    print(weights)

    # Plot cost history to visualize the training process
    plot_cost(cost_history,lr)

   
    test_predictions = np.dot(X_test,weights)
    mse_test = np.mean((test_predictions - y_test) ** 2)/2
    print(f"Mean squared error on test data: {mse_test}")


