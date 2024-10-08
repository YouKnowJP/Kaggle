#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:14:47 2024

@author: youknowjp
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_log_error
from xgboost import XGBRegressor

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

# Define the model
model = XGBRegressor(n_estimators=100, random_state=42)

# Create a pipeline that processes the data and then fits the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Define RMSLE function
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Make a scorer from the RMSLE function
rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

# Fit the pipeline on the entire training data
pipeline.fit(X, y)

# Evaluate the model using cross-validation with RMSLE
cv_scores = cross_val_score(pipeline, X, y, cv=3, scoring=rmsle_scorer)

# Use the fitted pipeline to make predictions on the test set
predictions = pipeline.predict(test_df.drop(['id'], axis=1))

# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Rings': predictions
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

# Calculate the mean of the cross-validation scores (RMSLE)
mean_rmsle = -cv_scores.mean()
print("Mean RMSLE:", mean_rmsle)
