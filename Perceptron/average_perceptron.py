import pandas as pd
import numpy as np
np.random.seed(32)

def averaged_perceptron_train(X_train, y_train, T=10, r=0.1):
   
    m, n = X_train.shape
    w = np.zeros(n)  # Initialize weight vector
    a = np.zeros(n)  # Initialize accumulated vector

    for epoch in range(T):
        indices = np.random.permutation(m)
        for i in indices:
            x_i = X_train[i]
            y_i = y_train[i]
            # Update rule if there is a misclassification
            if y_i * np.dot(w, x_i) <= 0:
                w = w + r * y_i * x_i
            # Accumulate the weight vector
            a = a + w
    return a,w

def perceptron_predict(X, a):
    return np.sign(np.dot(X, a))

def compute_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

if __name__ == "__main__":
    # Load data
    train_dataframe = pd.read_csv('bank-note/train.csv', header=None)
    test_dataframe = pd.read_csv('bank-note/test.csv', header=None)
    train_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_dataframe.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    # Prepare training and test sets
    X_train = train_dataframe.iloc[:, :-1].values
    y_train = train_dataframe.iloc[:, -1].values
    y_train[y_train == 0] = -1  # Convert labels to {-1, 1}
    X_test = test_dataframe.iloc[:, :-1].values
    y_test = test_dataframe.iloc[:, -1].values
    y_test[y_test == 0] = -1  # Convert labels to {-1, 1}
    
    # Train the Averaged Perceptron
    a,w= averaged_perceptron_train(X_train, y_train, T=10,r=0.01)
    
    # Predict on test data
    y_pred = perceptron_predict(X_test, a)
    
    # Compute average prediction error
    error = compute_error(y_test, y_pred)
    
    # Output the results to the console
    print("Finally Learned  weight vector:", w)
    print("Avrrage Weight is:",a)
    print("Average prediction error on test dataset:", error)
    
    # Save the results to a text file
    with open("averaged_perceptron_results.txt", "w") as file:
        file.write("Learned averaged weight vector: " + np.array2string(w) + "\n")
        file.write("Average prediction error on test dataset: " + str(error) + "\n")
