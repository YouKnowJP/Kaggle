#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:35:02 2024

@author: youknowjp
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE

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

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test_df.drop(columns=['id']))

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_scaled, y_encoded)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Function to perform grid search
def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

# LightGBM
lgbm_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'num_leaves': [31, 50, 100],
    'min_child_samples': [10, 20, 30],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [0.0, 0.1, 1.0]
}

lgbm_model = lgb.LGBMClassifier(random_state=42)
best_params_lgbm, best_score_lgbm = perform_grid_search(lgbm_model, lgbm_param_grid, X_train, y_train)

# XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [0.0, 0.1, 1.0],
    'gamma': [0.0, 0.1, 0.5],
    'min_child_weight': [1, 5, 10]
}

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
best_params_xgb, best_score_xgb = perform_grid_search(xgb_model, xgb_param_grid, X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2]
}
best_params_gb, best_score_gb = perform_grid_search(gb_model, gb_param_grid, X_train, y_train)

# AdaBoost
ada_model = AdaBoostClassifier(random_state=42)
ada_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
best_params_ada, best_score_ada = perform_grid_search(ada_model, ada_param_grid, X_train, y_train)

# Stacking Classifier
estimators = [
    ('lgbm', lgb.LGBMClassifier(**best_params_lgbm, random_state=42)),
    ('xgb', xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('gb', GradientBoostingClassifier(**best_params_gb, random_state=42)),
    ('ada', AdaBoostClassifier(**best_params_ada, random_state=42))
]
stacking_model = StackingClassifier(estimators=estimators, final_estimator=lgb.LGBMClassifier(random_state=42))

# Cross-validation with stacking model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stacking_scores = cross_val_score(stacking_model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"Cross-Validation Accuracy of the Stacking Model: {stacking_scores.mean()}")

# Train the stacking model on the full training data
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
classification_rep = classification_report(y_val, y_pred)

print(f"Validation Accuracy of the Stacking Model: {accuracy}")
print("Classification Report of the Stacking Model:")
print(classification_rep)

# Prepare the test data for prediction
X_test_scaled = scaler.transform(test_df.drop(columns=['id']))
test_ids = test_df['id']

# Make predictions on the test set
test_pred = stacking_model.predict(X_test_scaled)

# Create a DataFrame with 'id' and 'Target' columns
submission_df = pd.DataFrame({'id': test_ids, 'Target': label_encoder.inverse_transform(test_pred)})

# Save the predictions to a CSV file
submission_df.to_csv('test_predictions_stacking_model.csv', index=False)
