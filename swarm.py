import random
import pandas as pd
# Define data
data = pd.read_excel("C:/Users/warit/OneDrive/Desktop/ci4/AirQualityUCI.xlsx")

# Select and format the data
selected_features = data.iloc[:, [2, 5, 7, 9, 10, 11, 12, 13]].values
target = data.iloc[:, 4].values

# Standardize the data
def standardize_data(data):
    mean = sum(data) / len(data)
    std = (sum([(x - mean) ** 2 for x in data]) / len(data)) ** 0.5
    if std == 0:
        return [0.0 for _ in data]  # If std is zero, return a list of zeros to avoid division by zero
    else:
        return [(x - mean) / std for x in data]
    
# Define the neural network
def initialize_weights(input_size, hidden_layers, nodes, output_size):
    weights = []
    layer_size = input_size
    for _ in range(hidden_layers):
        weights.append([[random.uniform(-0.1, 0.1) for _ in range(layer_size)] for _ in range(nodes)])
        layer_size = nodes
    weights.append([[random.uniform(-0.1, 0.1) for _ in range(layer_size)] for _ in range(output_size)])
    return weights

def feedforward(inputs, weights):
    for layer_weights in weights:
        inputs = [sum([x * w for x, w in zip(inputs, node_weights)]) for node_weights in layer_weights]
        inputs = [max(0, x) for x in inputs]  # ReLU activation
    return inputs

# Define the loss function
def mean_absolute_error(y_true, y_pred):
    return sum([abs(true - pred) for true, pred in zip(y_true, y_pred)]) / len(y_true)

# Define the objective function for optimization
def objective_function(params, X, y):
    num_hidden_layers, num_nodes = params
    input_size = len(X[0])
    output_size = 1  # Since target is a single value
    weights = initialize_weights(input_size, num_hidden_layers, num_nodes, output_size)
    y_pred = [feedforward(x, weights)[0] for x in X]  # Access the first (and only) element
    loss = mean_absolute_error(y, y_pred)  # Calculate MAE directly without wrapping y
    return loss

# Optimization using random search
best_result = None
best_mae = float('inf')

# Number of random searches
num_random_searches = 100

for _ in range(num_random_searches):
    num_hidden_layers = random.randint(1, 5)
    num_nodes = random.randint(1, 20)
    params = (num_hidden_layers, num_nodes)
    mae = objective_function(params, selected_features, target)
    if mae < best_mae:
        best_mae = mae
        best_result = params

optimal_params = best_result

# Print the results
optimal_num_hidden_layers, optimal_num_nodes = optimal_params
print("Optimal Number of Hidden Layers:", optimal_num_hidden_layers)
print("Optimal Number of Nodes per Layer:", optimal_num_nodes)
print("Minimum MAE:", best_mae)

# Generate future data for 5 days and 10 days ahead
def generate_future_data(days_ahead):
    future_data = []
    last_data_point = selected_features[-1]
    for _ in range(days_ahead):
        future_data_point = [value + random.normalvariate(0, 1) for value in last_data_point]
        future_data.append(future_data_point)
        last_data_point = future_data_point

    return future_data

future_data_5_days = generate_future_data(5)
future_data_10_days = generate_future_data(10)

# Use the trained neural network to make predictions for the generated future data
optimal_weights = initialize_weights(len(selected_features[0]), optimal_num_hidden_layers, optimal_num_nodes, 1)
predicted_benzene_5_days = [feedforward(data_point, optimal_weights)[0] for data_point in future_data_5_days]
predicted_benzene_10_days = [feedforward(data_point, optimal_weights)[0] for data_point in future_data_10_days]

# Print or use the predicted results
print("Generated Future Data for 5 Days Ahead:")
print(future_data_5_days)

print("Generated Future Data for 10 Days Ahead:")
print(future_data_10_days)

print("Predicted Benzene Concentration for 5 Days Ahead:")
print(predicted_benzene_5_days)

print("Predicted Benzene Concentration for 10 Days Ahead:")
print(predicted_benzene_10_days)
