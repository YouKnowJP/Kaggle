#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 20:37:28 2024

@author: youknowjp
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import optuna

# Load Data
train_data = pd.read_csv("/Users/youknowjp/Downloads/playground-series-s4e6/train.csv")
test_data = pd.read_csv("/Users/youknowjp/Downloads/playground-series-s4e6/test copy.csv")

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
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
        'one_hot_max_size': trial.suggest_int('one_hot_max_size', 2, 10)
    }

    # Initialize CatBoost model
    model = CatBoostClassifier(**params, verbose=0)

    # Perform cross-validation
    score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    return score

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# Retrieve the best parameters
best_params = study.best_params
print(f"Best parameters: {best_params}")

# Train the model with the best parameters
best_model = CatBoostClassifier(**best_params, verbose=0)
best_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Calculate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy of CatBoost: {test_accuracy:.4f}")
