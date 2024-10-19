
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


# Function to calculate weighted entropy
def weighted_entropy(y, weights):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_weight = np.sum(weights)
    entropy_value = 0
    for i, cls in enumerate(unique_classes):
        class_weights = weights[y == cls]
        weighted_prob = np.sum(class_weights) / total_weight
        if weighted_prob > 0:
            entropy_value += -weighted_prob * np.log2(weighted_prob)
    
    return entropy_value

# Function to calculate weighted gini index
def weighted_gini_index(y, weights):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_weight = np.sum(weights)
    
    gini_value = 1
    for cls in unique_classes:
        class_weights = weights[y == cls]
        weighted_prob = np.sum(class_weights) / total_weight
        gini_value -= weighted_prob ** 2
    return gini_value


# Function to calculate weighted majority error
def weighted_majority_error(y, weights):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    total_weight = np.sum(weights)

    # Find the majority class by weight
    majority_class_weight = np.max([np.sum(weights[y == cls]) for cls in unique_classes])
    majority_error_value = 1 - (majority_class_weight / total_weight)
    
    return majority_error_value

# Generic function to calculate impurity
def calculate_impurity(y, weights, measure="entropy"):
    if measure == "entropy":
        return weighted_entropy(y,weights)
    elif measure == "gini":
        return weighted_gini_index(y,weights)
    elif measure == "majority_error":
        return weighted_majority_error(y,weights)
    else:
        raise ValueError(f"Unknown impurity measure: {measure}")

# Function to calculate weighted Information Gain
def weighted_info_gain(data, feature, label, weights, measure="entropy"):
   
    total_weight = np.sum(weights)
    
    total_impurity = calculate_impurity(data[label], weights, measure)
    
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_impurity = 0

    for value in values:
        subset = data[data[feature] == value]
        subset_weights = weights[data[feature] == value]
    
        impurity = calculate_impurity(subset[label], subset_weights, measure)
   
        weighted_impurity += (np.sum(subset_weights) / total_weight) * impurity

    information_gain = total_impurity - weighted_impurity
    return information_gain


# Function to find the best feature to split on
def best_feature(data, label, weights, measure="entropy"):
    features = data.columns[:-1]  # Select all features except the label
    information_gains = [weighted_info_gain(data, feature, label, weights, measure) for feature in features]
    
    # Return the feature with the highest information gain
    best_feature_index = np.argmax(information_gains)
    return features[best_feature_index]

def predict_single(tree, instance):
    if not isinstance(tree, dict):
        return tree  # Base case: if it's not a dictionary, it's a leaf node (prediction)
    
    feature = next(iter(tree))  # Get the feature at this node
    feature_value = instance[feature]
    
    if feature_value in tree[feature]:  # If the value exists in the subtree
        return predict_single(tree[feature][feature_value], instance)
    else:
        # If the feature value isn't found, return the majority class at this node
        return tree['majority_class']


# Function to make predictions using the decision tree
def predict_single(tree, instance):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))  
    feature_value = instance[feature]
    if feature_value in tree[feature]:      
        return predict_single(tree[feature][feature_value], instance)
    else:
        return tree['majority_class']

# Function to predict for a dataset
def predict(tree, data):
    predictions = data.apply(lambda row: predict_single(tree, row), axis=1)
    return predictions

# Function to calculate error rate
def calculate_error(predictions, true_labels):
    return np.mean(predictions != true_labels)

#Threeshold calculations for numerical 
def threshold_numerical_to_categorical(df, numerical_columns):
    for col in numerical_columns:
        col_threshold = df[col].median()
        df[col] = df[col].apply(lambda x: 1 if x >= col_threshold else 0)  
    return df


#function to load the bank data with numerical threshold 
def load_data_bank(file_path,columns):
    data=pd.read_csv(file_path)
    data.columns=columns
    features_labels=list(data.columns[:-1])
    class_label='label'
    non_categorical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    data=threshold_numerical_to_categorical(data,non_categorical_columns)
    feature_values=data.iloc[0: ,:-1]
    class_label_values=data.iloc[0:, -1]
    return data,features_labels,class_label,feature_values,class_label_values


def id3(data, original_data, features, label, weights, impurity_measure="entropy", max_depth=1, depth=0, parent_node_class=None):

    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]  # If all labels are the same, return the label

    elif len(data) == 0:
      
        return parent_node_class
    
    elif len(features) == 0 or depth >= max_depth:
        # If no features are left or max depth is reached, return the majority class
        return parent_node_class
    
    else:
        # Compute the majority class at the parent node
        parent_node_class = np.unique(data[label])[np.argmax(np.unique(data[label], return_counts=True)[1])]

        # Find the best feature to split the data using weighted information gain
        best_feat = best_feature(data, label, weights, impurity_measure)
    
        
        # Handle the case where no meaningful feature was found
        if best_feat is None:
            return parent_node_class
        
        # Create the root of the stump (feature to split on)
        tree = {best_feat: {}, 'majority_class': parent_node_class}
        
        # Remove the best feature from the set of features
        features = [feat for feat in features if feat != best_feat]
        
        # For each possible value of the best feature, create a branch
        for value in np.unique(data[best_feat]):
            sub_data = data[data[best_feat] == value]
            sub_weights = weights[data[best_feat] == value]
            
            # Recursively call id3 for the subtree
            subtree = id3(sub_data, data, features, label, sub_weights, impurity_measure, max_depth=1, depth=depth+1, parent_node_class=parent_node_class)
            tree[best_feat][value] = subtree
    
    return tree


def calculate_weighted_error(weights, predictions, true_labels):
    weighted_sum = np.sum(weights * true_labels * predictions)
    weighted_error = 0.5 - 0.5 * weighted_sum
    return weighted_error


def adaboost(data, feature_label, label, T,impurity_measure="gini"):
    weights = np.ones(len(data)) / len(data) 

    classifiers = []
    alphas = []
    imp=impurity_measure
  
    features_value = data.drop('label',axis=1)

    for t in range(T):
       
        stump = id3(data, data, feature_label, label, weights,imp, max_depth=1)
   
        predictions = predict(stump, features_value)

        predictions_converted=predictions.apply(lambda x: 1 if x == "no" else -1)
        predictions_converted=predictions_converted.to_numpy()
       
  
        data_converted=  data[label].apply(lambda x: 1 if x == "no" else -1)
        data_converted=data_converted.to_numpy()

    
        error = calculate_weighted_error(weights, predictions_converted, data_converted)
 
        alpha = 0.5 * np.log((1 - error) / (error +0.0000000001))

        weights = weights * np.exp(alpha * data_converted * predictions_converted)
        weights = weights / np.sum(weights)  

        classifiers.append(stump)
        alphas.append(alpha)

    return classifiers, alphas

def adaboost_predict(classifiers, alphas, data):
    final_predictions = np.zeros(len(data))
    
    for i in range(len(classifiers)):
        stump = classifiers[i]
        features_value = data.drop('label',axis=1)
        predictions = predict(stump, features_value)
        predictions_converted=predictions.apply(lambda x: 1 if x == "no" else -1)
        final_predictions += alphas[i] * predictions_converted
    return np.sign(final_predictions)

# Function to evaluate AdaBoost for a range of iterations
def evaluate_adaboost(data_train, data_test,feature_label, label, T_range,imp):
    train_errors = []
    test_errors = []

    classifiers, alphas = adaboost(data_train, feature_label, label,max(T_range),imp)
    data_train[label]= data_train[label].apply(lambda x: 1 if x == "no" else -1)
    data_test[label]= data_test[label].apply(lambda x: 1 if x == "no" else -1)

    for T in T_range:
        train_predictions = adaboost_predict(classifiers[:T], alphas[:T], data_train)
        test_predictions = adaboost_predict(classifiers[:T], alphas[:T], data_test)

        
        train_error = calculate_error(train_predictions, data_train[label])
       
        test_error = calculate_error(test_predictions, data_test[label])

        train_errors.append(train_error)
        test_errors.append(test_error)

    with open('adaboost_train_errors.txt', 'w') as f_train, open('adaboost_test_errors.txt', 'w') as f_test:
        for T, error in zip(T_range, train_errors):
            f_train.write(f"Num Trees: {T}, Training Error: {error}%\n")
        
        for T, error in zip(T_range, test_errors):
            f_test.write(f"Num Trees: {T}, Testing Error: {error}%\n")

    return train_errors, test_errors



# Function to plot training and test errors over iterations
def plot_errors(train_errors, test_errors, T_range):
    plt.figure(figsize=(10, 6)) 
    plt.plot(T_range, train_errors, label='Training Error', color='blue')
    plt.plot(T_range, test_errors, label='Test Error' ,color='red')
    plt.xlabel('Number of Iterations (T)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Training and Test Error Over Iterations in AdaBoost', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout()  
    plt.show()
    plt.savefig("Adaboost_Loss.png")


# Function to calculate the training and testing error of stumps at each iteration
def evaluate_stump_errors(data_train, data_test, feature_label, label,T_range, impurity_measure="entropy"):
    stump_train_errors = []
    stump_test_errors = []

    classifiers, _ = adaboost(data_train, feature_label, label, max(T_range), impurity_measure)
    features_value_train = data_train.drop('label',axis=1)
    features_value_test = data_test.drop('label',axis=1)
    for T in T_range:
        # Get the stump at the current iteration (T-1)
       
        stump_train_predictions = predict(classifiers[T-1] , features_value_train)
        stump_test_predictions =  predict(classifiers[T -1], features_value_test)
    
        
        stump_train_error = calculate_error(stump_train_predictions, data_train[label])
        stump_test_error = calculate_error(stump_test_predictions, data_test[label])

        stump_train_errors.append(stump_train_error)
        stump_test_errors.append(stump_test_error)
    
    with open('stump_train_errors.txt', 'w') as f_train, open('stump_test_errors.txt', 'w') as f_test:
        for T, error in zip(T_range, train_errors):
            f_train.write(f"Num Trees: {T}, Training Error: {error}%\n")
        
        for T, error in zip(T_range, test_errors):
            f_test.write(f"Num Trees: {T}, Testing Error: {error}%\n")
    return stump_train_errors, stump_test_errors

# Function to plot training and test errors for decision stumps
def plot_stump_errors(stump_train_errors, stump_test_errors, T_range):
    plt.figure(figsize=(10, 6))  
    plt.plot(T_range, stump_train_errors, label='Stump Training Error', color='green')
    plt.plot(T_range, stump_test_errors, label='Stump Test Error', color='orange')
    plt.xlabel('Number of Iterations (T)', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title('Training and Test Error for Decision Stumps Over Iterations', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.tight_layout() 
    plt.show()
    plt.savefig("Stump_Errors.png")


if __name__ == "__main__":
       training_file2='Data /bank/train.csv'
       testing_file2='Data /bank/test.csv'
       columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
       all_train,X_train_labels,y_train_label,X_train,y_train=load_data_bank(training_file2,columns)
       all_test,X_test_labels,y_test_label,X_test,y_test=load_data_bank(testing_file2,columns)
       T_range = range(1,500)
       impurity_measure="majority_error"

       train_errors, test_errors = evaluate_adaboost(all_train, all_test, X_train_labels,y_train_label, T_range,impurity_measure)
  
       plot_errors(train_errors, test_errors, T_range)

       stump_train_errors, stump_test_errors = evaluate_stump_errors(all_train, all_test, X_train_labels, y_train_label, T_range, impurity_measure)

       plot_stump_errors(stump_train_errors, stump_test_errors, T_range)


            
