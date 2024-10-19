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

def optimize_weights(x, y):
    return np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))



if __name__=="__main__":
    training_file = "Concrete/train.csv"
    testing_file = "Concrete/test.csv"
    column_names = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'slump']
    X_train, y_train, X_test, y_test = get_data(training_file, testing_file, column_names)

   

    # Train model using SGD
    weights = optimize_weights(X_train, y_train)
    print("Final Weights for training are:")
    print(weights)
  
    test_predictions = np.dot(X_test, weights)
    mse_test = np.mean((test_predictions - y_test) ** 2) / 2
    print(f"Mean squared error on test data: {mse_test}")
