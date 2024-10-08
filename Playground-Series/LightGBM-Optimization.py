#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 22:48:46 2024

@author: youknowjp
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import optuna

# Load the datasets
train_df = pd.read_csv('C:/Users/thong/Downloads/Playground-Series/train.csv')
test_df = pd.read_csv('C:/Users/thong/Downloads/Playground-Series/test.csv')

# Inspect the unique values in the target column
unique_values = train_df['Target'].unique()
print(f'Unique values in target column before mapping: {unique_values}')

# Create a mapping dictionary
value_mapping = {
    'Dropout': 0,
    'Enrolled': 1,
    'Graduate': 2
}

# Map the target column to numeric values
train_df['Target'] = train_df['Target'].map(value_mapping)

# Drop rows with missing target values after mapping
train_df = train_df.dropna(subset=['Target'])

# Verify the unique values after mapping
print(train_df['Target'].unique())

# Split the data into features and target
X = train_df.drop(columns=['Target'])
y = train_df['Target']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    param = {
        'objective': 'multiclass',
        'num_class': 3,
        'boosting_type': 'gbdt',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
    }
    
    # Create a LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    # Train the model
    model = lgb.train(param, train_data, valid_sets=[valid_data], num_boost_round=1000)
    
    # Predict on the validation set
    y_pred_proba = model.predict(X_valid)
    y_pred = y_pred_proba.argmax(axis=1)
    accuracy = accuracy_score(y_valid, y_pred)
    
    return accuracy

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Train the final model with the best hyperparameters
best_params = study.best_params
best_params['objective'] = 'multiclass'
best_params['num_class'] = 3
best_params['metric'] = 'multi_logloss'
best_params['verbosity'] = -1

train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

model = lgb.train(best_params, train_data, valid_sets=[valid_data], num_boost_round=1000)

# Predict on the validation set
y_pred_proba = model.predict(X_valid)
y_pred = y_pred_proba.argmax(axis=1)
accuracy = accuracy_score(y_valid, y_pred)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f'Validation Accuracy: {accuracy}')
print(f'Validation RMSE: {rmse}')
