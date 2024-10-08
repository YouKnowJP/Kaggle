#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:53:54 2024

@author: youknowjp
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

# Load data
train_df = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/train.csv')
test_df = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/test.csv')

# Feature Engineering
train_df['Volume'] = train_df['Length'] * train_df['Diameter'] * train_df['Height']
test_df['Volume'] = test_df['Length'] * test_df['Diameter'] * test_df['Height']

# Define features and target
X = train_df.drop(['id', 'Rings'], axis=1)
y = np.log1p(train_df['Rings'])  # Log transform of target to reduce skewness and improve RMSLE

# Preprocessing for numerical data
numerical_features = ['Length', 'Diameter', 'Height', 'Whole weight', 'Whole weight.1', 'Whole weight.2', 'Shell weight', 'Volume']
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

# Preprocessing for categorical data
categorical_features = ['Sex']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42))
]

# Create the stacking ensemble
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=LinearRegression(),
    passthrough=True
)

# Create a pipeline that processes the data and then fits the model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('feature_selection', SelectFromModel(GradientBoostingRegressor(n_estimators=50))),
    ('model', stack)
])

# Fit the pipeline on the entire training data
pipeline.fit(X, y)

# Use the fitted pipeline to make predictions on the test set
predictions = pipeline.predict(test_df.drop(['id'], axis=1))
predictions = np.expm1(predictions)  # Inverse of log1p

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Rings': predictions
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

# Define RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.expm1(y_true), np.expm1(y_pred)))

# Evaluate the model using RMSLE
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
rmsle_score = rmsle_scorer(pipeline, X, y)
print("RMSLE on training set:", rmsle_score)
