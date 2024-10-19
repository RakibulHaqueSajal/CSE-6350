
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt



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



#Defining the training function for bagged tree learning
def bagged_tree_train(data, features, label, num_trees=10,impurity_measure="entropy",max_depth=5):
    trees = []
    for t in range(num_trees):
        bootstrap_indices = np.random.choice(range(len(data)), size=len(data), replace=True)
        bootstrap_sample = data.iloc[bootstrap_indices]
        tree = id3(bootstrap_sample, bootstrap_sample, features,label,impurity_measure=impurity_measure, max_depth=max_depth) 
        trees.append(tree)

    return trees

def majority_vote(predictions):
    vote_counts = {}
    for pred in predictions:
        if pred not in vote_counts:
            vote_counts[pred] = 1
        else:
            vote_counts[pred] += 1
    # Return the prediction with the highest count
    return max(vote_counts, key=vote_counts.get)

# Bagged tree prediction function with majority voting
def bagged_tree_predict(trees, data):
    
    tree_predictions = []
    for tree in trees:
        predictions = predict(tree, data)
        tree_predictions.append(predictions)

    tree_predictions = np.array(tree_predictions).T

    final_predictions = [majority_vote(instance_preds) for instance_preds in tree_predictions]
    
    return np.array(final_predictions)

# Function to evaluate and plot the error for varying number of trees
def plot_error_vs_num_trees(data, features, label, X_train, y_train, X_test, y_test, max_trees=500, impurity_measure="entropy", max_depth=5):
    train_errors = []
    test_errors = []
    num_trees_range = range(1, max_trees + 1)

    with open('bagged_trees_training_errors.txt', 'w') as f_train, open('bagged_trees_testing_errors.txt', 'w') as f_test:
        for num_trees in num_trees_range:
            trees = bagged_tree_train(data, features, label, num_trees=num_trees, impurity_measure=impurity_measure, max_depth=max_depth)
          
            y_train_pred = bagged_tree_predict(trees, X_train)
            y_test_pred = bagged_tree_predict(trees, X_test)

           
            train_error = calculate_error(y_train_pred, y_train) 
            test_error = calculate_error(y_test_pred, y_test) 

            train_errors.append(train_error)
            test_errors.append(test_error)

            f_train.write(f'Num Trees: {num_trees}, Training Error: {train_error}%\n')
            f_test.write(f'Num Trees: {num_trees}, Testing Error: {test_error}%\n')
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(num_trees_range, train_errors, label='Training Error', color='blue')
    plt.plot(num_trees_range, test_errors, label='Test Error', color='red')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error Rate')
    plt.title('Training and Test Error vs. Number of Trees in Bagged Classifier')
    plt.legend()
    plt.grid(True)
    plt.savefig("Bagged_Training_and_Testing_Loss.png")
    plt.show()



if __name__ == "__main__":
  
    dataset = "bank"
    if dataset == "bank":
        training_file2 = 'Data /bank/train.csv'
        testing_file2 = 'Data /bank/test.csv'
        columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact',
                   'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
        all_train, X_train_labels, y_train_label, X_train, y_train = load_data_bank(training_file2, columns)
        all_test, X_test_labels, y_test_label, X_test, y_test = load_data_bank(testing_file2, columns)

    plot_error_vs_num_trees(all_train, X_train_labels, y_train_label, X_train, y_train, X_test, y_test, max_trees=500, impurity_measure="entropy", max_depth=16)
  

         

    

        
                


    
