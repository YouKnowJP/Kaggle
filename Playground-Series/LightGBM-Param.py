#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:19:35 2024

@author: youknowjp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import early_stopping, log_evaluation, LGBMClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import set_config
import optuna
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100)
set_config(transform_output='pandas')
pd.options.mode.chained_assignment = None

# Load Data
train = pd.read_csv("C:/Users/thong/Downloads/Playground-Series/train.csv")
test = pd.read_csv("C:/Users/thong/Downloads/Playground-Series/test.csv")
train.Target.value_counts().plot(kind='bar')

class Model:
    def __init__(self, train, test):
        self.train = train
        self.test = test
        self.model_dict = dict()
        self.test_predict_list = list()

    def preprocess(self):
        cat_features = ['Marital status', 'Application mode', 'Course',
                        'Previous qualification', 'Nacionality', "Mother's qualification",
                        "Father's qualification", "Mother's occupation",
                        "Father's occupation"]
        for feature in cat_features:
            dtype = pd.CategoricalDtype(categories=list(set(self.train[feature]) | set(self.test[feature])), ordered=False)
            for df in [self.train, self.test]:
                df[feature] = df[feature].astype(dtype)
        self.train.Target = self.train.Target.map({"Graduate": 0,
                                                   "Dropout": 1,
                                                   "Enrolled": 2})
        self.train['Application mode'] = self.train['Application mode'].replace({12: np.NaN, 4: np.NaN, 35: np.NaN, 9: np.NaN, 3: np.NaN})
        self.test['Application mode'] = self.test['Application mode'].replace({14: np.NaN, 35: np.NaN, 19: np.NaN, 3: np.NaN})
        self.train['Course'] = self.train['Course'].replace({979: np.NaN, 39: np.NaN})
        self.test['Course'] = self.test['Course'].replace({7500: np.NaN, 9257: np.NaN, 2105: np.NaN, 4147: np.NaN})
        self.train['Previous qualification'] = self.train['Previous qualification'].replace({37: np.NaN, 36: np.NaN, 17: np.NaN, 11: np.NaN})
        self.test['Previous qualification'] = self.test['Previous qualification'].replace({17: np.NaN, 11: np.NaN, 16: np.NaN})

    def fit(self, params, name):
        target_col = ['Target']
        drop_col = ['id']
        self.preprocess()

        train_cols = [col for col in self.train.columns.to_list() if col not in target_col + drop_col]
        scores = list()

        for i in range(1):
            mskf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=22)
            oof_valid_preds = np.zeros(self.train[train_cols].shape[0])

            for fold, (train_idx, valid_idx) in enumerate(mskf.split(self.train[train_cols], self.train[target_col])):
                X_train, y_train = self.train[train_cols].iloc[train_idx], self.train[target_col].iloc[train_idx]
                X_valid, y_valid = self.train[train_cols].iloc[valid_idx], self.train[target_col].iloc[valid_idx]

                if name == 'lgbm':
                    early_stopping_callback = early_stopping(100, first_metric_only=True, verbose=False)
                    verbose_callback = log_evaluation(125)
                    algo = LGBMClassifier(random_state=i+fold, **params)
                    algo.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                             callbacks=[early_stopping_callback, verbose_callback], eval_metric='multi_logloss')

                valid_preds = algo.predict(X_valid)

                oof_valid_preds[valid_idx] = valid_preds
                test_predict = algo.predict(self.test[train_cols])

                self.test_predict_list.append(test_predict)

                score = accuracy_score(y_valid, valid_preds)
                print(f"\nFold: {fold+1}\n===>Accuracy score: {score}")
                # Calculating the confusion matrix
                cm = confusion_matrix(y_valid, valid_preds, labels=[x for x in range(0, 3)])

                # Displaying the confusion matrix
                disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                              display_labels=[x for x in range(0, 3)])
                disp.plot()
                plt.show()
                self.model_dict[f'fold_{fold}'] = algo

            oof_score = accuracy_score(self.train[target_col], oof_valid_preds)
            print(f"The OOF accuracy score for iteration {i+1} is {oof_score}")
            scores.append(oof_score)
        
        final_accuracy = np.mean(scores)
        return final_accuracy

    def predict_and_save(self, params, name, output_file):
        self.fit(params, name)
        
        # Majority voting for test predictions
        predictions_3d = np.array(self.test_predict_list)
        final_predictions = []
        for entry_predictions in predictions_3d.T:  # Iterate over each entry's predictions
            unique, counts = np.unique(entry_predictions, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        # Save predictions to CSV
        submission = pd.DataFrame({'id': self.test['id'], 'Target': final_predictions})
        submission['Target'] = submission['Target'].map({0: "Graduate", 1: "Dropout", 2: "Enrolled"})
        submission.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

# Optuna optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 10000),
        'num_class': 3,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'verbosity': -1,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 50),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'min_data_per_groups': trial.suggest_int('min_data_per_groups', 1, 100),
    }

    model = Model(train, test)
    accuracy = model.fit(params, 'lgbm')
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params = trial.params

model = Model(train, test)
accuracy = model.fit(best_params, 'lgbm')
print(f"Overall accuracy score: {accuracy}")

# model.predict_and_save(best_params, 'lgbm', 'submission.csv')
