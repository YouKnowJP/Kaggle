#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:34:02 2024

@author: youknowjp
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_log_error

# Load data
train_df = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/train.csv')
test_df = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/test.csv')

# Define features and target
X = train_df.drop(['id', 'Rings'], axis=1)
y = train_df['Rings']

# Preprocessing for numerical data
numerical_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 'Whole weight.2', 'Shell weight']
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_features = ['Sex']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define multiple models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
}

# Fit models and collect predictions
predictions = []
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X, y)
    pred = pipeline.predict(test_df.drop(['id'], axis=1))
    predictions.append(pred)

# Average predictions
final_predictions = np.mean(predictions, axis=0)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Rings': final_predictions
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

# Evaluate RMSLE for each model using cross-validation
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    rmsle_scorer = make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred)), greater_is_better=False)
    cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring=rmsle_scorer)
    mean_rmsle = -cv_scores.mean()
    print(f"Mean RMSLE for {name}: {mean_rmsle}")
