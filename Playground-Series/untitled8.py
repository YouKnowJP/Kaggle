# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:51:41 2024

@author: thong
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

import lightgbm as lgb

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

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

params = {'n_estimators': 8000, 
          'num_class': 3,
          'boosting_type': 'gbdt',
          'objective': 'multiclass',
          'metric': 'multi_logloss',
          'verbosity': -1,
          'random_state': 99, 
          'reg_alpha': 1.7878527151970849, 
          'reg_lambda': 1.391543710164331, 
          'colsample_bytree': 0.5, 
          'subsample': 0.5, 
          'learning_rate': 0.04, 
          'max_depth': 20, 
          'num_leaves': 70, 
          'min_child_samples': 40, 
          'min_data_per_groups': 16
         }

def cross_val_train(X, y, test, params):
    spl = 10  # Number of folds

    # Initialize arrays with predictions and oof predictions
    test_preds = np.zeros((len(test), 3))
    val_preds = np.zeros((len(X), 3))
    val_scores = []

    # Perform cross-validation split
    cv = KFold(spl, shuffle=True, random_state=42)

    # "for" cycle to train for each fold
    for fold, (train_ind, valid_ind) in enumerate(cv.split(X, y)):
        # Divide train and validation data
        X_train = X.iloc[train_ind]
        y_train = y[train_ind]
        X_val = X.iloc[valid_ind]
        y_val = y[valid_ind]

        # Initiate model lightGBM
        model = lgb.LGBMClassifier(**params)

        # Fit the model
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(stopping_rounds=70), lgb.log_evaluation(100)])

        # Predictions on train and validation data
        y_pred_trn = model.predict_proba(X_train)
        y_pred_val = model.predict_proba(X_val)

        # Compute accuracy
        train_acc = accuracy_score(y_train, np.argmax(y_pred_trn, axis=1))
        val_acc = accuracy_score(y_val, np.argmax(y_pred_val, axis=1))

        # Print partial results for the fold
        print("Fold:", fold, " Train Accuracy:", np.round(train_acc, 5), " Val Accuracy:", np.round(val_acc, 5))

        # Compute test predictions and oof predictions
        test_preds += model.predict_proba(test[features]) / spl
        val_preds[valid_ind] = model.predict_proba(X_val)
        val_scores.append(val_acc)
        print("-" * 50)

    return val_scores, val_preds, test_preds

val_scores, val_preds, test_preds = cross_val_train(X, y, test, params)

val_preds_out = np.argmax(val_preds, axis=1)
print(accuracy_score(y, val_preds_out))

# Submission
y_test = np.argmax(test_preds, axis=1)
y_test = label_encoder.inverse_transform(y_test)

# Create a new submission DataFrame
submission = pd.DataFrame({'id': test['id'], 'Target': y_test})

# Save the submission DataFrame to a new CSV file
submission.to_csv('C:/Users/thong/Downloads/Playground-Series/submission_new.csv', index=False)
