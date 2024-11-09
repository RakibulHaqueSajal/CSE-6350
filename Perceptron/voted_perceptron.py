import pandas as pd
import numpy as np
import random
np.random.seed(32)

def voted_perceptron_train(X_train, y_train, T=10, r=0.1):
    m, n = X_train.shape
    w = np.zeros(n)  # Initialize the first weight vector
    weight_vectors = []  # List to store (w_m, C_m) pairs
    c_m = 1  # Initialize count for the first weight vector

    for epoch in range(T):
        indices = np.random.permutation(m)
        for i in indices:
            x_i = X_train[i]
            y_i = y_train[i]
            # Check for misclassification
            if y_i * np.dot(w, x_i) <= 0:
                # Store the current weight vector and its count
                weight_vectors.append((w.copy(), c_m))
                # Update weight vector
                w = w + r * y_i * x_i
                c_m = 1  # Reset count for the new weight vector
            else:
                # Increment count if correctly classified
                c_m += 1

    # Add the last weight vector and its count
    weight_vectors.append((w.copy(), c_m))
    return weight_vectors,w

def voted_perceptron_predict(X, weight_vectors):
    predictions = []
    for x in X:
        vote = sum(c_m * np.sign(np.dot(w_m, x)) for w_m, c_m in weight_vectors)
        predictions.append(np.sign(vote))
    return np.array(predictions)

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
    
    # Train the Voted Perceptron
    weight_vectors,w = voted_perceptron_train(X_train, y_train, T=10)
    
    # Predict on test data
    y_pred = voted_perceptron_predict(X_test, weight_vectors)
    
    # Compute average prediction error
    error = compute_error(y_test, y_pred)
    
    # Output the results to the console
    print("Final Learned weight vector:", w)
    print("Average prediction error on test dataset:", error)
    
    # Save the results to a text file
    with open("voted_perceptron_results.txt", "w") as file:
        file.write("Learned weight vectors and counts:\n")
        for w, c in weight_vectors:
            file.write("w: " + np.array2string(w) + ", count: " + str(c) + "\n")
        file.write("\nAverage prediction error on test dataset: " + str(error) + "\n")
