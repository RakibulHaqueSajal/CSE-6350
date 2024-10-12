

import pandas as pd
import numpy as np
import argparse

# Function to parse the arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Decision Tree Im')
    parser.add_argument('dataset', metavar='D', type=str, default="car",
                        help='3 choices: car, bank, bank_missing')
    return parser.parse_args()


# Function to load the CSV data
def load_data(file_path,columns):
    data=pd.read_csv(file_path)
    data.columns=columns
    features_labels=list(data.columns[:-1])
    class_label='label'
    feature_values=data.iloc[0: ,:-1]
    class_label_values=data.iloc[0:, -1]
    return data,features_labels,class_label,feature_values,class_label_values

#function to calculate the entropy 
def entropy(y):
    elements, counts = np.unique(y, return_counts=True)
    entropy_value = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy_value

# Function to calculate Gini Index
def gini_index(y):
    elements, counts = np.unique(y, return_counts=True)
    gini = 1 - np.sum([(counts[i]/np.sum(counts))**2 for i in range(len(elements))])
    return gini

# Function to calculate Majority Error
def majority_error(y):
    elements, counts = np.unique(y, return_counts=True)
    majority_class_count = np.max(counts)
    majority_error_value = 1 - (majority_class_count / np.sum(counts))
    return majority_error_value

# Generic function to calculate impurity
def calculate_impurity(y, measure="entropy"):
    if measure == "entropy":
        return entropy(y)
    elif measure == "gini":
        return gini_index(y)
    elif measure == "majority_error":
        return majority_error(y)
    else:
        raise ValueError(f"Unknown impurity measure: {measure}")

# Function to calculate Information Gain (using any impurity measure)
def info_gain(data, feature, label, measure="entropy"):
    total_impurity = calculate_impurity(data[label], measure)
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_impurity = np.sum([(counts[i]/np.sum(counts)) * calculate_impurity(data.where(data[feature] == values[i]).dropna()[label], measure) for i in range(len(values))])
    information_gain = total_impurity - weighted_impurity
    return information_gain

# Function to find the best feature to split on
def best_feature(data, label, measure="entropy"):
    features = data.columns[:-1]
    information_gains = [info_gain(data, feature, label, measure) for feature in features]
    best_feature_index = np.argmax(information_gains)
    return features[best_feature_index]

# Recursive function to build the ID3 tree with maximum depth and majority class at each node
def id3(data, original_data, features, label, impurity_measure="entropy", max_depth=np.inf, depth=0, parent_node_class=None):

    if len(np.unique(data[label])) <= 1:
        return np.unique(data[label])[0]
    elif len(data) == 0:
        return np.unique(original_data[label])[np.argmax(np.unique(original_data[label], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    elif depth >= max_depth:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[label])[np.argmax(np.unique(data[label], return_counts=True)[1])]
        best_feat = best_feature(data, label, impurity_measure)
        tree = {best_feat: {}, 'majority_class': parent_node_class}
        features = [feat for feat in features if feat != best_feat]

        for value in np.unique(data[best_feat]):
            sub_data = data.where(data[best_feat] == value).dropna()
            subtree = id3(sub_data, original_data, features, label, impurity_measure, max_depth, depth + 1, parent_node_class)
            tree[best_feat][value] = subtree

    return tree

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
import pandas as pd

#Threeshold calculations for numerical 
def threshold_numerical_to_categorical(df, numerical_columns):
    for col in numerical_columns:
        col_threshold = df[col].median()
        df[col] = df[col].apply(lambda x: 1 if x >= col_threshold else 0)  
    return df

#Unknown Replacement 
def replace_unknown_with_majority(df, columns):
    for col in columns:
        # Calculate the majority value (mode) of the column, excluding 'unknown'
        majority_value = df[df[col] != 'unknown'][col].mode()[0]
        # Replace 'unknown' with the majority value
        df[col] = df[col].replace('unknown', majority_value)
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
    
#function to load the bank data with missing value median 
def load_data_bank_missing(file_path,columns):
    data=pd.read_csv(file_path)
    data.columns=columns
    features_labels=list(data.columns[:-1])
    class_label='label'
    non_categorical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    data=threshold_numerical_to_categorical(data,non_categorical_columns)
    missing_columns = ['job', 'education', 'contact', 'poutcome']
    data=replace_unknown_with_majority(data,missing_columns)
    feature_values=data.iloc[0: ,:-1]
    class_label_values=data.iloc[0:, -1]
    return data,features_labels,class_label,feature_values,class_label_values


#Evaluate the decision Tree
def evaluate_decision_tree(all_train, X_train_lables, y_train_lable,X_train,y_train,X_test,y_test, impurity_measures, max_depth_range):
   
    results_table = pd.DataFrame(columns=["Impurity Measure", "Depth", "Training Error (%)", "Test Error (%)"])
    
    for impurity_measure in impurity_measures:
       
        for depth in max_depth_range:
            print(f"Evaluating tree with depth {depth} and impurity measure '{impurity_measure}'...")
            
            tree = id3(all_train, all_train, X_train_labels, y_train_lable, impurity_measure=impurity_measure, max_depth=depth)
            
            y_train_pred = predict(tree, X_train)
            
            y_test_pred = predict(tree, X_test)
            
            train_error = calculate_error(y_train_pred, y_train)
            
            test_error = calculate_error(y_test_pred, y_test)
            
            row = pd.DataFrame({
                "Impurity Measure": [impurity_measure],
                "Depth": [depth],
                "Training Error (%)": [train_error * 100],
                "Test Error (%)": [test_error * 100]
            })
            results_table = pd.concat([results_table, row], ignore_index=True)
    
    return results_table


if __name__ == "__main__":
    args = parse_arguments()
    dataset = str(args.dataset)
    impurity_measures = ["entropy", "gini", "majority_error"]
    if dataset=="car":
        columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
        training_file="Data /car/train.csv"
        testing_file="Data /car/test.csv"
        all_train,X_train_labels,y_train_label,X_train,y_train=load_data(training_file,columns)
        all_test,X_test_labels,y_test_label,X_test,y_test=load_data(testing_file,columns)
        depth_range = range(1, 7)
        results_table = evaluate_decision_tree(all_train, X_train_labels, y_train_label,X_train,y_train, X_test, y_test, impurity_measures, depth_range)
        print("\nResults Table (Training and Test Errors for Different Depths and Impurity Measures):")
        print(results_table)
    elif dataset=="bank":
       training_file2='Data /bank/train.csv'
       testing_file2='Data /bank/train.csv'
       columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
       all_train,X_train_labels,y_train_label,X_train,y_train=load_data_bank(training_file2,columns)
       all_test,X_test_labels,y_test_label,X_test,y_test=load_data_bank(testing_file2,columns)
       depth_range = range(1, 16)
       results_table = evaluate_decision_tree(all_train, X_train_labels, y_train_label,X_train,y_train, X_test, y_test, impurity_measures, depth_range)
       print("\nResults Table (Training and Test Errors for Different Depths and Impurity Measures):")
       print(results_table)
    elif dataset=="bank_missing":
       training_file2='Data /bank/train.csv'
       testing_file2='Data /bank/train.csv'
       columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
       all_train,X_train_labels,y_train_label,X_train,y_train=load_data_bank_missing(training_file2,columns)
       all_test,X_test_labels,y_test_label,X_test,y_test=load_data_bank_missing(testing_file2,columns)

       depth_range = range(1, 16)
       results_table = evaluate_decision_tree(all_train, X_train_labels, y_train_label,X_train,y_train, X_test, y_test, impurity_measures, depth_range)
       print("\nResults Table (Training and Test Errors for Different Depths and Impurity Measures):")
       print(results_table)
    results_table.to_csv('Errors.csv')




         

    

        
                


    
