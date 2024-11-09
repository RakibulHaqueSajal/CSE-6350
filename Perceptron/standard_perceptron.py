import pandas as pd
import numpy as np
import random
random.seed(32)

def perceptron_train(X_train, y_train, T=10, r=0.1):
    m, n = X_train.shape
    w = np.zeros(n)  # Initialize weight vector

    for epoch in range(T):
        indices = np.random.permutation(m)
        for i in indices:
            x_i = X_train[i]
            y_i = y_train[i]
            # Update rule if there is a misclassification
            if y_i * np.dot(w, x_i) <= 0:
                w = w + r * y_i * x_i
    return w

def perceptron_predict(X, w):
    return np.sign(np.dot(X, w))

def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

if __name__ == "__main__":
    train_dataframe = pd.read_csv('bank-note/train.csv', header=None)
    test_dataframe = pd.read_csv('bank-note/test.csv', header=None)
    train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    X_train = train_dataframe.iloc[:, :-1].values
    y_train = train_dataframe.iloc[:, -1].values
    y_train[y_train == 0] = -1
    X_test = test_dataframe.iloc[:, :-1].values
    y_test = test_dataframe.iloc[:, -1].values
    y_test[y_test == 0] = -1
    
    # Train the Perceptron
    w = perceptron_train(X_train, y_train, T=10)
    
    # Predict on test data
    y_pred = perceptron_predict(X_test, w)
    
    # Compute average prediction error
    error = compute_error(y_test, y_pred)
    
    # Output the results to the console
    print("Learned weight vector:", w)
    print("Average prediction error on test dataset:", error)
    
    # Save the results to a text file
    with open("perceptron_results.txt", "w") as file:
        file.write("Learned weight vector: " + np.array2string(w) + "\n")
        file.write("Average prediction error on test dataset: " + str(error) + "\n")
