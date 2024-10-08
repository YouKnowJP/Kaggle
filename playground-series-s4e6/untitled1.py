#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:14:34 2024

@author: youknowjp
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the data files
train_data_path = '/Users/youknowjp/Downloads/playground-series-s4e6/train.csv'
test_data_path = '/Users/youknowjp/Downloads/playground-series-s4e6/test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Split the dataset into features and target
X = train_df.drop(columns=['id', 'Target'])
y = train_df['Target']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the LightGBM model
lgbm_model = LGBMClassifier(random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize Grid Search
grid_search = GridSearchCV(estimator=lgbm_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best Parameters: {best_params}")
print(f"Best Cross-Validation Accuracy: {best_score}")

# Train the model with the best parameters
best_lgbm_model = LGBMClassifier(**best_params, random_state=42)
best_lgbm_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_best_lgbm = best_lgbm_model.predict(X_val)

# Evaluate the model
accuracy_best_lgbm = accuracy_score(y_val, y_pred_best_lgbm)
classification_rep_best_lgbm = classification_report(y_val, y_pred_best_lgbm)

print(f"Validation Accuracy: {accuracy_best_lgbm}")
print("Classification Report:")
print(classification_rep_best_lgbm)

# Prepare the test data for prediction
X_test = test_df.drop(columns=['id'])
test_ids = test_df['id']

# Make predictions on the test set
test_pred_best_lgbm = best_lgbm_model.predict(X_test)

# Create a DataFrame with 'id' and 'Target' columns
submission_df = pd.DataFrame({'id': test_ids, 'Target': label_encoder.inverse_transform(test_pred_best_lgbm)})

# Save the predictions to a CSV file
submission_df.to_csv('test_predictions_best_lgbm.csv', index=False)
