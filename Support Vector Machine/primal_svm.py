import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
 # Define the stochastic subgradient descent for SVM with two learning rate schedules
def svm_sgd(X, y, C, gamma0, a, T, N, schedule=1):
     m, n = X.shape
     w = np.zeros(n)
     errors_train = []
     errors_test = []
        
     for epoch in range(T):
            # Choose the learning rate schedule
         if schedule == 1:
                # Schedule 1: gamma_t = gamma0 / (1 + (gamma0 / a) * t)
            gamma_t = gamma0 / (1 + (gamma0 / a) * epoch)
         elif schedule == 2:
                # Schedule 2: gamma_t = gamma0 / (1 + t)
            gamma_t = gamma0 / (1 + epoch)
            
            # Shuffle training data
         indices = np.random.permutation(m)
         X_shuffled = X[indices]
         y_shuffled = y[indices]
            
         for i in range(m):
            xi = X_shuffled[i]
            yi = y_shuffled[i]
                
                # Compute the hinge loss subgradient
            if yi * np.dot(w, xi) < 1:
                    # Misclassified or within margin
                w = w - gamma_t * w + gamma_t * C * N * yi * xi  # Updated weight rule considering m
            else:
                    # Correctly classified
                w = (1 - (gamma_t / N)) * w
            
            # Calculate training and test error
         train_error = np.mean(np.sign(np.dot(X, w)) != y)
         test_error = np.mean(np.sign(np.dot(X_test, w)) != y_test)
            
         errors_train.append(train_error)
         errors_test.append(test_error)
        
     return w, errors_train, errors_test

    

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
# Create a shared folder to save all figures
output_folder = "svm_plots"
os.makedirs(output_folder, exist_ok=True)

# Hyperparameters
gamma0 = 0.01  # Initial learning rate
a = 0.001  # Tuning parameter for learning rate schedule
C_values = [100 / 873, 500 / 873, 700 / 873]
T = 100  # Maximum number of epochs
N = X_train.shape[0]  # Number of training samples

# Open the file to save results
# Assuming svm_sgd function is already defined as per your previous code

# Create a shared folder to save all figures
output_folder = "svm_plots"
os.makedirs(output_folder, exist_ok=True)

# Hyperparameters
gamma0 = 0.01  # Initial learning rate
a = 0.001  # Tuning parameter for learning rate schedule
C_values = [100 / 873, 500 / 873, 700 / 873]
T = 100  # Maximum number of epochs
N = X_train.shape[0]  # Number of training samples

# Create a shared folder to save all figures
output_folder = "svm_plots"
os.makedirs(output_folder, exist_ok=True)

# Hyperparameters
gamma0 = 0.01  # Initial learning rate
a = 0.001  # Tuning parameter for learning rate schedule
C_values = [100 / 873, 500 / 873, 700 / 873]
T = 100  # Maximum number of epochs
N = X_train.shape[0]  # Number of training samples

# Open the file to save results
with open('primal_svm_errors.txt', 'w') as f_results:
    f_results.write('C, Learning Rate Schedule, Final Training Error, Final Test Error, Weights\n')

    # Loop through C values and both learning rate schedules
    for C in C_values:
        # Create a figure for each value of C and two learning rate schedules
        plt.figure(figsize=(12, 6))

        # Loop for two learning rate schedules
        for schedule in [1, 2]:
            # Train the model for the given C and schedule
            w, errors_train, errors_test = svm_sgd(X_train, y_train, C, gamma0, a, T, N, schedule=schedule)

            # Get the final errors
            final_train_error = errors_train[-1]
            final_test_error = errors_test[-1]

            # Convert the learned weights to space-separated string with 5 decimal places
            weights_str = " ".join([f"{w_i:.5f}" for w_i in w])

            # Plot training and test errors for the current schedule
            schedule_name = f'Schedule {schedule}'
            plt.plot(range(T), errors_train, label=f'Training error ({schedule_name})')
            plt.plot(range(T), errors_test, label=f'Test error ({schedule_name})', linestyle='--')

            # Write the results for this schedule
            f_results.write(f'{C:.5f}, Schedule {schedule}, {final_train_error:.5f}, {final_test_error:.5f}, {weights_str}\n')

        # Set the title, labels, and legend for each plot
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title(f'Training and Test Errors for C = {C:.5f} with Learning Rate Schedules')
        plt.legend()

        # Save the plot for the current C value and both schedules
        plot_filename = os.path.join(output_folder, f'C_{C:.5f}_errors.png')
        plt.savefig(plot_filename)  # Save the figure as a .png file in the shared folder

        # Close the plot after saving to avoid memory overload
        plt.close()

