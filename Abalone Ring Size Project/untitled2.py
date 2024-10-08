#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:32:05 2024

@author: youknowjp
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
import os
import lightgbm as lgb
import xgboost as xgb
import catboost as cb


# Load data from local files
train = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/train.csv')
test = pd.read_csv('/Users/youknowjp/Downloads/Abalone Ring Size Project/test.csv')

categorical_features = ['Sex']
numeric_features = [col for col in train.columns if col not in categorical_features + ['Rings']]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

base_models = [
    ('lgbm', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror')),
    ('catboost', cb.CatBoostRegressor(n_estimators=100, learning_rate=0.05, verbose=0))
]

stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(),
    passthrough=True
)

model = Pipeline([
    ('preprocessor', preprocessor),
    ('stacker', stack_model)
])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])

for fold, (idx_tr, idx_va) in enumerate(kf.split(train, train['Rings'])):
    X_tr, X_va = train.iloc[idx_tr], train.iloc[idx_va]
    y_tr, y_va = train['Rings'].iloc[idx_tr], train['Rings'].iloc[idx_va]

    model.fit(X_tr, y_tr)
    oof_preds[idx_va] = model.predict(X_va)
    test_preds += model.predict(test) / kf.n_splits

score = mean_squared_log_error(train['Rings'], oof_preds, squared=False)
print(f"OOF RMSLE: {score:.5f}")


# Create a DataFrame for submission
submission = pd.DataFrame({
    'id': test['id'],
    'Rings': test_preds
})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)
