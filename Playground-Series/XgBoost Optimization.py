#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 18:27:48 2024

@author: youknowjp
"""

# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load Data
train_data = pd.read_csv("/Users/youknowjp/Downloads/playground-series-s4e6/train.csv")
test_data = pd.read_csv("/Users/youknowjp/Downloads/playground-series-s4e6/test.csv")

# Encode the 'Target' variable
label_encoder = LabelEncoder()
train_data['Target_encoded'] = label_encoder.fit_transform(train_data['Target'])

# Define features and target
features = train_data.drop(['id', 'Target', 'Target_encoded'], axis=1)
target = train_data['Target_encoded']

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42, stratify=target)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Define the preprocessing pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocess the data
X_train_scaled = pipeline.fit_transform(X_train)
X_valid_scaled = pipeline.transform(X_valid)
X_test_scaled = pipeline.transform(X_test)

# Apply SMOTE to balance the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define the objective function for Optuna
# Adjust cv number and n_trials for better results
def objective(trial):
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),  # L2 regularization
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0)    # L1 regularization
    }

    model = xgb.XGBClassifier(**param)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy').mean()
    return accuracy

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1, n_jobs=-1)

# Retrieve the best parameters and best accuracy
best_params = study.best_params
best_accuracy = study.best_value

# Train the model with the best parameters
best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
best_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = best_model.predict(X_test_scaled)

# Calculate test set accuracy
test_accuracy = accuracy_score(y_test, y_pred)

# Function to return best parameters and accuracies
def get_best_params_and_accuracies():
    return best_params, best_accuracy, test_accuracy

# Retrieve and print best parameters and accuracies
best_params, best_accuracy, test_accuracy = get_best_params_and_accuracies()

print(f"Best parameters: {best_params}")
print(f"Best cross-validated accuracy: {best_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
