#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:27:48 2024

@author: youknowjp
"""

# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna

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

# Define the objective function for Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 1e-3, 10, log=True),
        'alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
        'early_stopping_rounds': 50
    }

    model = xgb.XGBClassifier(**param, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train_scaled, y_train, eval_set=[(X_valid_scaled, y_valid)], verbose=False)

    y_valid_pred = model.predict(X_valid_scaled)
    accuracy = accuracy_score(y_valid, y_valid_pred)

    return accuracy

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Get the best parameters
best_params = study.best_params
best_params.pop('early_stopping_rounds')  # Remove early_stopping_rounds from the best_params
print(f"Best parameters: {best_params}")

# Train the model with the best parameters
best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', early_stopping_rounds=50)
best_model.fit(X_train_scaled, y_train, eval_set=[(X_valid_scaled, y_valid)], verbose=False)

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Calculate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy of XGBoost: {test_accuracy:.4f}")

# Function to return best parameters and accuracies
def get_best_params_and_accuracies():
    return best_params, None, test_accuracy

# Retrieve and print best parameters and accuracies
best_params, _, test_accuracy = get_best_params_and_accuracies()
print(f"Best parameters: {best_params}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Preprocess and predict on the provided test.csv
# Drop the 'id' column and standardize the test data
test_features = test_data.drop(['id'], axis=1)
test_features_scaled = scaler.transform(test_features)

# Make predictions
test_predictions_encoded = best_model.predict(test_features_scaled)

# Convert encoded predictions back to original labels
test_predictions = label_encoder.inverse_transform(test_predictions_encoded)

# Create a DataFrame with the 'id' and 'Target' columns
output_df = pd.DataFrame({'id': test_data['id'], 'Target': test_predictions})

# Save the predictions to a CSV file
output_df.to_csv("/Users/youknowjp/Downloads/playground-series-s4e6/test copy.csv", index=False)
print("Predictions saved to predictions.csv")
