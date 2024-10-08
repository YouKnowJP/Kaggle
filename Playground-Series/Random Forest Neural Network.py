# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load Data
train_data = pd.read_csv("C:/Users/thong/Downloads/Playground-Series/train.csv")
test_data = pd.read_csv("C:/Users/thong/Downloads/Playground-Series/test.csv")

# Encode the 'Target' variable
label_encoder = LabelEncoder()
train_data['Target_encoded'] = label_encoder.fit_transform(train_data['Target'])

# Define features and target
features = train_data.drop(['id', 'Target', 'Target_encoded'], axis=1)
target = train_data['Target_encoded']

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_valid_tensor = torch.tensor(X_valid_scaled, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define the objective function for Optuna
def objective(trial):
    input_size = X_train_tensor.shape[1]
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    num_classes = len(label_encoder.classes_)
    
    model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    model.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    accuracy = correct / total
    return accuracy

# Create a study object and optimize the objective function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")

# Train the model with the best parameters
input_size = X_train_tensor.shape[1]
hidden_size = best_params['hidden_size']
learning_rate = best_params['learning_rate']
num_classes = len(label_encoder.classes_)

best_model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)

num_epochs = 50
for epoch in range(num_epochs):
    best_model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = best_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Evaluate on the test set
best_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = best_model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_accuracy = correct / total
print(f"Test accuracy of Neural Network: {test_accuracy:.4f}")

# Implementing Random Forest Model

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Predict on the validation set
y_valid_pred = rf_model.predict(X_valid_scaled)

# Calculate the validation accuracy
rf_valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Validation accuracy of Random Forest: {rf_valid_accuracy:.4f}")

# Predict on the test set
y_test_pred = rf_model.predict(X_test_scaled)

# Calculate the test accuracy
rf_test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy of Random Forest: {rf_test_accuracy:.4f}")

# Function to return best parameters and accuracies
def get_best_params_and_accuracies():
    return best_params, study.best_trial.value, test_accuracy, rf_valid_accuracy, rf_test_accuracy

# Retrieve and print best parameters and accuracies
best_params, best_validation_accuracy, nn_test_accuracy, rf_valid_accuracy, rf_test_accuracy = get_best_params_and_accuracies()
print(f"Best parameters for Neural Network: {best_params}")
print(f"Best validation accuracy for Neural Network: {best_validation_accuracy:.4f}")
print(f"Test accuracy of Neural Network: {nn_test_accuracy:.4f}")
print(f"Validation accuracy of Random Forest: {rf_valid_accuracy:.4f}")
print(f"Test accuracy of Random Forest: {rf_test_accuracy:.4f}")
