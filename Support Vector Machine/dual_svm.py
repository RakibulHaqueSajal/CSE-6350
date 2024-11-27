import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def gaussian_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float):
     # Compute the squared Euclidean distance
    squared_distance = np.sum((x1 - x2) ** 2)
    return np.exp(-squared_distance / gamma)

def objective_function(alpha, x, y, kernel_type, gamma, c):
    # Compute the kernel matrix based on the kernel type
    if kernel_type == "linear":
        kernel_matrix = linear_kernel(x, x)  # Linear kernel
    elif kernel_type == "gaussian":
        kernel_matrix = gaussian_kernel(x, x, gamma)  # Gaussian kernel
    term_1 = -np.sum(alpha)
    term_2 = 0.5 * np.dot(alpha, np.dot(np.outer(y, y), np.dot(kernel_matrix, alpha)))
    return term_1 + term_2

def constraints(alpha, y):
    return np.dot(alpha.T, y)

def fit_svm(x, y, kernel_type, C, gamma=0.0):
    n_samples, n_features = x.shape
    alpha = np.zeros(n_samples)
    cons = ({'type': 'eq', 'fun': constraints, 'args': (y,)})
    bounds=[(0, C)]
    initial_guess = np.zeros(n_samples)
    solution = minimize(fun=objective_function, x0=initial_guess, bounds=bounds,
                        method='SLSQP', constraints=cons, args=(x, y, kernel_type, gamma, C))
    alpha = solution.x
    support_vectors = np.where((alpha > 1e-5) & (alpha < C))[0]

    w = np.dot(alpha * y, x)
    b = np.dot(alpha, y)

    return alpha, support_vectors, w, b, kernel_type

def predict_svm(x, support_vectors, alpha, y, x_train, kernel_type,gamma=0.0):
    prediction_res = []
    for i in range(len(x)):
        if kernel_type=="linear":
           kernel=linear_kernel(x_train[support_vectors], x[i])
        elif kernel_type=="gaussian":
           kernel=gaussian_kernel(x_train[support_vectors],x[i],gamma)
        
        prediction = np.sign(np.sum(alpha[support_vectors] * y[support_vectors] *
                                    kernel))
        if prediction > 0:
            prediction_res.append(1)
        else:
            prediction_res.append(-1)
    return np.array(prediction_res)

def evaluate_svm(x, y, support_vectors, alpha, y_train, x_train, kernel,gamma=0.0):
    predictions = predict_svm(x, support_vectors, alpha, y_train, x_train, kernel,gamma)
    return np.mean(predictions != y)
def track_fixed_support_vectors(x_train, y_train, C, gamma_list, kernel_type="gaussian"):
    
    # Dictionary to store support vectors for each gamma value
    support_vectors_for_gamma = {}

    # Iterate through each gamma value and fit the SVM
    for gamma in gamma_list:
        # Fit the SVM for the current gamma value
        alpha, support_vectors, w, b, kernel = fit_svm(x_train, y_train, kernel_type, C, gamma)
        
        # Store the support vectors for the current gamma
        support_vectors_for_gamma[gamma] = set(support_vectors)
        #print(f"Support vectors for C={C} and gamma={gamma}: {support_vectors}")

    # Dictionary to store the common support vectors for each pair of gamma values
    common_support_vectors = {}

    # Compare support vectors between all pairs of gamma values
    for gamma1 in gamma_list:
        for gamma2 in gamma_list:
            if gamma1 != gamma2:
                # Find the intersection (common support vectors) between the two gamma values
                common_support_vectors_pair = support_vectors_for_gamma[gamma1].intersection(support_vectors_for_gamma[gamma2])
                
                # Store the common support vectors for the pair of gamma values
                common_support_vectors[(gamma1, gamma2)] = common_support_vectors_pair
                print(f"Common support vectors for gamma={gamma1} and gamma={gamma2}: {common_support_vectors_pair}")
                print(f"Number of common support vectors: {len(common_support_vectors_pair)}")

    # Return the support vectors and common support vectors
    return support_vectors_for_gamma, common_support_vectors

if __name__ == "__main__":
    # Read the training and test datasets
    train_df = pd.read_csv('bank-note/train.csv', header=None)
    test_df = pd.read_csv('bank-note/test.csv', header=None)

    # Assign column names to the data
    train_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    # Extract features and labels
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    # Convert labels to {1, -1}
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1

    # Hyperparameters
    C_values = [100 / 873, 500 / 873, 700 / 873]
    gamma_list = [0.1, 0.5, 1, 5, 100]

    # Open files to save results
    with open('dual_svm_linear.txt', 'w') as f_linear, open('dual_svm_gaussian.txt', 'w') as f_gaussian:
        f_linear.write('C\ttraining_error\ttesting_error\tweights\n')
        f_gaussian.write('C\tgamma\ttraining_error\ttesting_error\tweights\tnum_support_vectors\n')
        for C in C_values:
            alpha, support_vectors, w, b, kernel = fit_svm(X_train, y_train, "linear", C)
            linear_training_error = evaluate_svm(X_train, y_train, support_vectors, alpha, y_train, X_train, kernel)
            linear_testing_error = evaluate_svm(X_test, y_test, support_vectors, alpha, y_train, X_train, kernel)
            print(f'Train Error: {linear_training_error}')
            print(f'Test Error: {linear_testing_error}')
            print("Number of support vectors: " + str(len(support_vectors)))
            f_linear.write(f"{C}\t{linear_training_error}\t{linear_testing_error}\t{w}\n")

        for C in C_values:
            for gamma in gamma_list:
                alpha, support_vectors, w, b, kernel = fit_svm(X_train, y_train, "gaussian", C, gamma)
                gaussian_training_error = evaluate_svm(X_train, y_train, support_vectors, alpha, y_train, X_train, kernel,gamma)
                gaussian_testing_error =evaluate_svm(X_test, y_test, support_vectors, alpha, y_train, X_train, kernel,gamma)
                print(f'C: {C}, gamma: {gamma}')
                print(f'Train Error: {gaussian_training_error}')
                print(f'Test Error: {gaussian_testing_error}')
                print("Number of support vectors: " + str(len(support_vectors)))
                f_gaussian.write(f"{C}\t{gamma}\t{gaussian_training_error}\t{gaussian_testing_error}\t{np.append(w, b)}\t{len(support_vectors)}\n")

gamma_list_2 = [0.01, 0.1, 0.5]  
C=500 / 873

# Assuming fit_svm and other necessary functions are defined elsewhere
support_vectors_for_gamma, common_support_vectors = track_fixed_support_vectors(X_train,y_train, C, gamma_list_2)

with open('dual_svm_fixed_support_vectors.txt', 'w') as f:
    for gamma, support_vectors in support_vectors_for_gamma.items():
        f.write(f"Gamma: {gamma}, Support Vectors: {support_vectors}\n")
    for (gamma1, gamma2), common_sv in common_support_vectors.items():
        f.write(f"Common Support Vectors between gamma={gamma1} and gamma={gamma2}: {common_sv}\n")