# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:00:29 2024

@author: thong
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import optuna

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)

# Load Data
train = pd.read_csv("C:/Users/thong/Downloads/Playground-Series/train.csv")
test = pd.read_csv("C:/Users/thong/Downloads/Playground-Series/test.csv")

features = train.columns.drop(['id', 'Target'])

X_test = test[features]
X = train[features]
y = train['Target']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 20),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'min_data_per_groups': trial.suggest_int('min_data_per_groups', 1, 50),
        'num_class': 3,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'random_state': 42
    }

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(50)])
    
    y_pred_val = model.predict_proba(X_val)
    val_acc = accuracy_score(y_val, np.argmax(y_pred_val, axis=1))

    return val_acc

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, n_jobs=-1)  # Enable parallel optimization

# Print the best parameters
print("Best hyperparameters:", study.best_params)

# Train the final model using the best parameters
best_params = study.best_params
best_params['num_class'] = 3
best_params['boosting_type'] = 'gbdt'
best_params['objective'] = 'multiclass'
best_params['metric'] = 'multi_logloss'
best_params['verbosity'] = -1
best_params['random_state'] = 42

# Perform final training with cross-validation
def cross_val_train(X, y, test, params, n_splits=10):
    test_preds = np.zeros((len(test), 3))
    val_preds = np.zeros((len(X), 3))
    val_scores = []

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_ind], y[train_ind]
        X_val, y_val = X.iloc[valid_ind], y[valid_ind]

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False), lgb.log_evaluation(100)])

        y_pred_val = model.predict_proba(X_val)
        
        val_acc = accuracy_score(y_val, np.argmax(y_pred_val, axis=1))
        print(f"Fold: {fold} Validation Accuracy: {val_acc:.5f}")
        
        test_preds += model.predict_proba(test[features]) / n_splits
        val_preds[valid_ind] = y_pred_val
        val_scores.append(val_acc)
        print("-" * 50)

    return val_scores, val_preds, test_preds

val_scores, val_preds, test_preds = cross_val_train(X, y, X_test, best_params)

val_preds_out = np.argmax(val_preds, axis=1)
final_accuracy = accuracy_score(y, val_preds_out)
print(f"Final accuracy: {final_accuracy:.5f}")

# Return the best parameters
print(f"Best parameter: {best_params}")
