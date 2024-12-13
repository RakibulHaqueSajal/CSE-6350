import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, depth, width, activation_fn, init_fn):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(depth):
            in_features = input_size if i == 0 else width
            out_features = width if i < depth - 1 else 1
            layer = nn.Linear(in_features, out_features)
            init_fn(layer.weight)
            layers.append(layer)
            if i < depth - 1:  # Add activation between hidden layers
                layers.append(activation_fn)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()

def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, header=None)
    test_df = pd.read_csv(test_path, header=None)
    train_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
    test_df.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']

    train_x = torch.tensor(train_df.iloc[:, :-1].values, dtype=torch.float32)
    train_y = torch.tensor(train_df.iloc[:, -1].values, dtype=torch.float32)
    test_x = torch.tensor(test_df.iloc[:, :-1].values, dtype=torch.float32)
    test_y = torch.tensor(test_df.iloc[:, -1].values, dtype=torch.float32)

    return train_x, train_y, test_x, test_y

def train_model(model, train_x, train_y, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

def evaluate_model(model, x, y):
    model.eval()
    with torch.no_grad():
        outputs = model(x)
        predictions = (outputs >= 0.5).float()
        error = (predictions != y).float().mean().item()
    return error


if __name__ == "__main__":
    train_path = '/Users/mdrakibulhaque/Desktop/Neural Network/bank-note/train.csv'
    test_path = '/Users/mdrakibulhaque/Desktop/Neural Network/bank-note/test.csv'

    train_x, train_y, test_x, test_y = load_data(train_path, test_path)

    depths = [3, 5, 7,]
    widths = [5, 25,50,100]
    epochs = 50
    lr = 0.0001

    results = []

    with open("pytorch_results.txt", "w") as result_file:
        result_file.write("Depth\tWidth\tActivation\tTrain_Error\tTest_Error\n")

        for depth in depths:
            for width in widths:
                for activation_name, activation_fn, init_fn in [
                    ("tanh", nn.Tanh(), nn.init.xavier_uniform_),
                    ("relu", nn.ReLU(), nn.init.kaiming_uniform_)
                ]:
                    model = NeuralNetwork(train_x.shape[1], depth, width, activation_fn, init_fn)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.BCEWithLogitsLoss()

                    train_model(model, train_x, train_y, optimizer, criterion, epochs)

                    train_error = round(evaluate_model(model, train_x, train_y), 9)
                    test_error = round(evaluate_model(model, test_x, test_y), 9)

                    results.append((depth, width, activation_name, train_error, test_error))

                    result_file.write(f"{depth}\t{width}\t{activation_name}\t{train_error}\t{test_error}\n")
                    print(f"Depth={depth}, Width={width}, Activation={activation_name}, Train Error={train_error}, Test Error={test_error}")
