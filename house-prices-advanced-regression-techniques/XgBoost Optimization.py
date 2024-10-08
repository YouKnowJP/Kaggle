#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 01:49:42 2024

@author: youknowjp
"""

# Load libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load the train and test data
train_df = pd.read_csv('/Users/youknowjp/Downloads/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/Users/youknowjp/Downloads/house-prices-advanced-regression-techniques/test.csv')

# Handle missing values for numerical features using SimpleImputer
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns
num_features = num_features.drop('SalePrice')  # Remove target column from features
imputer = SimpleImputer(strategy='median')
train_df[num_features] = imputer.fit_transform(train_df[num_features])
test_df[num_features] = imputer.transform(test_df[num_features])

# Handle missing values for categorical features using SimpleImputer
cat_features = train_df.select_dtypes(include=['object']).columns
imputer = SimpleImputer(strategy='most_frequent')
train_df[cat_features] = imputer.fit_transform(train_df[cat_features])
test_df[cat_features] = imputer.transform(test_df[cat_features])

# Encode categorical variables using LabelEncoder
encoders = {}
for feature in cat_features:
    encoder = LabelEncoder()
    train_df[feature] = encoder.fit_transform(train_df[feature])
    test_df[feature] = encoder.transform(test_df[feature])
    encoders[feature] = encoder

# Split data into features and target
X = train_df.drop(['Id', 'SalePrice'], axis=1)
y = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    param = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5)
    }
    
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50, verbose=False)
    
    preds = model.predict(X_valid)
    mse = mean_squared_error(np.log1p(y_valid), np.log1p(preds))
    return mse

# Create the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print the best parameters and the best score (MSE)
best_params = study.best_params
best_mse = study.best_value
print(f"Best parameters: {best_params}")
print(f"Best MSE: {best_mse}")
