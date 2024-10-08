#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:59:20 2024

@author: youknowjp
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load data from local files
train = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/train.csv')
test = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/test.csv')


# Define the preprocessor
categorical_features = ['Sex']
numeric_features = [col for col in train.columns if col not in categorical_features + ['Rings']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, min_samples_leaf=8, max_features='auto'))
])

# Prepare for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
oof_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])

# Cross-validation
for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train['Rings'])):
    X_tr, X_va = train.iloc[idx_tr], train.iloc[idx_va]
    y_tr, y_va = train['Rings'].iloc[idx_tr], train['Rings'].iloc[idx_va]

    model.fit(X_tr, y_tr)
    oof_preds[idx_va] = model.predict(X_va)
    test_preds += model.predict(test) / kf.n_splits

# Evaluate OOF predictions
score = mean_squared_log_error(train['Rings'], oof_preds, squared=False)
print(f"OOF RMSLE: {score:.5f}")

# Create submission
submission = pd.DataFrame({'Rings': test_preds}, index=test["id"])
submission_filename = 'submission.csv'
submission.to_csv(submission_filename)

# Optionally print the first few lines of the submission file
print(submission.head())
