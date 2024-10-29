#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:06:00 2024

@author: youknowjp
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from typing import Any, Dict, List, Optional, Tuple
from functools import partial
from contextlib import contextmanager

import optuna
import lightgbm as lgb
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import torch

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer  # Using SimpleImputer

# ==========================
# 1. Configuration
# ==========================

# Define file paths (Ensure TRAIN_PATH points to train.csv and TEST_PATH to test.csv)
TRAIN_PATH = '/kaggle/input/playground-series-s4e10/train.csv'
TEST_PATH = '/kaggle/input/playground-series-s4e10/test.csv'
SAMPLE_SUBMISSION_PATH = '/kaggle/input/playground-series-s4e10/sample_submission.csv'

# Define target variable
TARGET = 'loan_status'

# Define output directories
OUTPUT_DIR = 'output'
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
PARAMS_DIR = os.path.join(OUTPUT_DIR, 'params')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create output directories if they don't exist
os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PARAMS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================
# 2. Data Loading
# ==========================

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and testing datasets.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Loaded training and testing DataFrames.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print(f"Training data shape: {train.shape}")
    print(f"Testing data shape: {test.shape}")
    return train, test

# ==========================
# 3. Data Preprocessing
# ==========================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features to enhance model performance.

    Args:
        df (pd.DataFrame): DataFrame to perform feature engineering on.

    Returns:
        pd.DataFrame: DataFrame with new features.
    """
    # Basic Feature Transformations
    df['income_to_age'] = df['person_income'] / (df['person_age'] + 1)
    df['loan_to_income'] = df['loan_amnt'] / (df['person_income'] + 1)
    df['rate_to_loan'] = df['loan_int_rate'] / (df['loan_amnt'] + 1)
    df['age_squared'] = df['person_age'] ** 2
    df['log_income'] = np.log1p(df['person_income'])
    df['age_credit_history_interaction'] = df['person_age'] * df['cb_person_cred_hist_length']
    
    # Categorical Binning (if applicable)
    if 'person_age' in df.columns:
        df['age_category'] = pd.cut(df['person_age'], bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elder'])
    if 'person_income' in df.columns:
        df['income_category'] = pd.qcut(df['person_income'], q=5, 
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Binary Features
    df['high_loan_to_income'] = (df['loan_percent_income'] > 0.5).astype(int)
    df['is_new_credit_user'] = (df['cb_person_cred_hist_length'] < 2).astype(int)
    df['high_interest_rate'] = (df['loan_int_rate'] > df['loan_int_rate'].mean()).astype(int)
    df['high_loan_amount'] = (df['loan_amnt'] > df['loan_amnt'].quantile(0.75)).astype(int)
    df['intent_home_match'] = ((df['loan_intent'] == 'HOMEIMPROVEMENT') & 
                                (df['person_home_ownership'] == 'OWN')).astype(int)
    df['high_risk_flag'] = ((df['loan_percent_income'] > 0.4) &
                            (df['loan_int_rate'] > df['loan_int_rate'].mean()) &
                            (df['cb_person_default_on_file'] == 'Y')).astype(int)
    
    # Interaction Features
    if 'loan_intent' in df.columns and 'loan_grade' in df.columns:
        df['intent_grade_interaction'] = df['loan_intent'].astype(str) + '_' + df['loan_grade'].astype(str)
    if 'person_home_ownership' in df.columns and 'loan_intent' in df.columns:
        df['home_ownership_intent'] = df['person_home_ownership'].astype(str) + '_' + df['loan_intent'].astype(str)
    if 'cb_person_default_on_file' in df.columns and 'loan_grade' in df.columns:
        df['default_grade_interaction'] = df['cb_person_default_on_file'].astype(str) + '_' + df['loan_grade'].astype(str)
    if 'loan_amount_category' in df.columns and 'person_home_ownership' in df.columns:
        df['home_ownership_loan_interaction'] = df['person_home_ownership'].astype(str) + '_' + df['loan_amount_category'].astype(str)
    if 'cb_person_default_on_file' in df.columns and 'loan_int_rate' in df.columns:
        df['default_rate_interaction'] = df['cb_person_default_on_file'].astype(str) + '_' + \
                                         pd.cut(df['loan_int_rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']).astype(str)
    if 'person_age' in df.columns and 'loan_int_rate' in df.columns:
        df['age_interest_interaction'] = df['person_age'] * df['loan_int_rate']
    
    # Ratio Features
    df['loan_to_employment'] = df['loan_amnt'] / (df['person_emp_length'] + 1)
    df['credit_history_to_age'] = df['cb_person_cred_hist_length'] / (df['person_age'] + 1)
    df['rate_to_credit_history'] = df['loan_int_rate'] / (df['cb_person_cred_hist_length'] + 1)
    df['creditworthiness_score'] = (df['person_income'] / (df['loan_amnt'] * df['loan_int_rate'])) * \
                                   (df['cb_person_cred_hist_length'] + 1)
    df['age_to_employment'] = df['person_age'] / (df['person_emp_length'] + 1)
    df['rate_to_age'] = df['loan_int_rate'] / (df['person_age'] + 1)
    df['income_to_loan'] = df['person_income'] / (df['loan_amnt'] + 1)
    
    # Polynomial Features
    num_features = ['person_age', 'person_income', 'person_emp_length', 
                    'loan_amnt', 'loan_int_rate', 'loan_percent_income']
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(df[num_features])

    try:
        poly_feature_names = poly.get_feature_names_out(num_features)
    except AttributeError:
        poly_feature_names = poly.get_feature_names(num_features)

    for i, name in enumerate(poly_feature_names):
        if '1 ' in name:
            continue  # Skip original features
        df[f'poly_{name}'] = poly_features[:, i]

    # Trigonometric Features
    df['age_sin'] = np.sin(2 * np.pi * df['person_age'] / 100)
    df['age_cos'] = np.cos(2 * np.pi * df['person_age'] / 100)
    
    # Stability Score
    if all(col in df.columns for col in ['person_emp_length', 'person_income', 'loan_amnt', 'cb_person_cred_hist_length']):
        df['stability_score'] = (df['person_emp_length'] * df['person_income']) / \
                                 (df['loan_amnt'] * (df['cb_person_cred_hist_length'] + 1))
    
    # Additional Features
    if 'loan_grade' in df.columns:
        grade_mapping = {'A':5, 'B':4, 'C':3, 'D':2, 'E':1, 'F':0, 'G':0}
        df['risk_score'] = df['loan_percent_income'] * df['loan_int_rate'] * \
                           (5 - df['loan_grade'].map(grade_mapping))
    if 'loan_intent' in df.columns and 'loan_amnt' in df.columns:
        df['normalized_loan_amount'] = df.groupby('loan_intent')['loan_amnt'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
    if 'loan_amnt' in df.columns:
        df['loan_amount_category'] = pd.qcut(df['loan_amnt'], q=5, 
                                            labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
    
    return df

def preprocess_data(train: pd.DataFrame, test: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Handle missing values, encode categorical variables, and perform feature engineering.

    Args:
        train (pd.DataFrame): Training DataFrame.
        test (pd.DataFrame): Testing DataFrame.
        target (str): Target variable name.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed training and testing DataFrames.
    """
    # Combine train and test for consistent preprocessing
    combined = pd.concat([train.drop(columns=[target], errors='ignore'), test], axis=0, ignore_index=True)
    print(f"Combined data shape before preprocessing: {combined.shape}")

    # Feature Engineering
    combined = feature_engineering(combined)

    # Identify numerical and categorical columns
    numerical_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = combined.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode categorical variables using Label Encoding
    for col in categorical_cols:
        combined[col] = combined[col].astype('category').cat.codes

    # Handle missing values using Simple Imputer with mean strategy
    imputer = SimpleImputer(strategy='mean')
    combined[numerical_cols] = imputer.fit_transform(combined[numerical_cols])

    # Split back into train and test
    train_preprocessed = combined.iloc[:train.shape[0], :].copy()
    test_preprocessed = combined.iloc[train.shape[0]:, :].copy()

    # Add target variable back to training data
    train_preprocessed[target] = train[target].values

    print(f"Training data shape after preprocessing: {train_preprocessed.shape}")
    print(f"Testing data shape after preprocessing: {test_preprocessed.shape}")

    return train_preprocessed, test_preprocessed

# ==========================
# 4. Suppress Warnings Context Manager
# ==========================

@contextmanager
def suppress_stderr():
    """
    Context manager to suppress stderr output.
    """
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ==========================
# 5. Model Definitions (Adjusted for GPU)
# ==========================

class ModelBase:
    """
    Base class for models with Optuna hyperparameter optimization.
    """
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, target: str, 
                 categorical_feats: List[str], base_params: Optional[Dict[str, Any]] = None):
        self.train = train
        self.test = test
        self.target = target
        self.categorical_feats = categorical_feats
        self.base_params = base_params if base_params is not None else {}
        self.model_dict: Dict[str, Any] = {}
        self.test_predict_list: List[np.ndarray] = []
    
    def fit(self, params: Dict[str, Any]) -> Tuple[List[float], List[np.ndarray], np.ndarray]:
        """
        Train the model using Stratified K-Fold cross-validation.

        Args:
            params (Dict[str, Any]): Hyperparameters for the model.

        Returns:
            Tuple[List[float], List[np.ndarray], np.ndarray]: ROC AUC scores, test predictions, and out-of-fold predictions.
        """
        label = self.target
        features = [col for col in self.train.columns if col != label]
        X = self.train[features]
        y = self.train[label].values
        test_X = self.test[features]
        
        scores = []
        mskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(self.train))
        
        for fold, (train_idx, valid_idx) in enumerate(mskf.split(X, y)):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
            
            model = self.get_model(params)
            
            with suppress_stderr():
                model.fit(
                    X_train, y_train, 
                    eval_set=[(X_valid, y_valid)],
                )
            
            valid_pred = model.predict_proba(X_valid)[:, 1]
            oof_preds[valid_idx] = valid_pred
            score = roc_auc_score(y_valid, valid_pred)
            scores.append(score)
            
            test_pred = model.predict_proba(test_X)[:, 1]
            self.test_predict_list.append(test_pred)
            self.model_dict[f'fold_{fold}'] = model
            
            print(f"Fold {fold + 1} ROC AUC: {score:.4f}")
        
        oof_score = roc_auc_score(y, oof_preds)
        print(f"Overall ROC AUC: {oof_score:.4f}\n")
        scores.append(oof_score)
        
        return scores, self.test_predict_list, oof_preds
    
    def get_model(self, params: Dict[str, Any]) -> Any:
        """
        Instantiate the model. To be implemented by subclasses.

        Args:
            params (Dict[str, Any]): Hyperparameters for the model.

        Returns:
            Any: Instantiated model.
        """
        raise NotImplementedError
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization. To be implemented by subclasses.

        Args:
            trial (optuna.Trial): Optuna trial object.

        Returns:
            float: Mean ROC AUC score.
        """
        raise NotImplementedError
    
    def optimize(self, n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.

        Args:
            n_trials (int, optional): Number of Optuna trials. Defaults to 100.

        Returns:
            Dict[str, Any]: Best hyperparameters found by Optuna.
        """
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        return study.best_params

class Model_gbdt(ModelBase):
    def get_model(self, params: Dict[str, Any]) -> lgb.LGBMClassifier:
        params = {**self.base_params, **params}
        params['device'] = 'gpu'  # Enable GPU
        return lgb.LGBMClassifier(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

class Model_goss(ModelBase):
    def get_model(self, params: Dict[str, Any]) -> lgb.LGBMClassifier:
        params = {**self.base_params, **params}
        params['boosting_type'] = 'goss'
        params['device'] = 'gpu'  # Enable GPU
        return lgb.LGBMClassifier(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

class Model_loss(ModelBase):
    def get_model(self, params: Dict[str, Any]) -> CatBoostClassifier:
        params = {**self.base_params, **params}
        params['task_type'] = 'GPU'  # Enable GPU
        return CatBoostClassifier(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-4, 1.0),
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

class Model_sym(ModelBase):
    def get_model(self, params: Dict[str, Any]) -> CatBoostClassifier:
        params = {**self.base_params, **params}
        params['task_type'] = 'GPU'  # Enable GPU
        return CatBoostClassifier(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-4, 1.0),
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

class Model_depth(ModelBase):
    def get_model(self, params: Dict[str, Any]) -> CatBoostClassifier:
        params = {**self.base_params, **params}
        params['task_type'] = 'GPU'  # Enable GPU
        return CatBoostClassifier(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            'iterations': trial.suggest_int('iterations', 100, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 1e-4, 1.0),
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

class Model_gbtree(ModelBase):
    def get_model(self, params: Dict[str, Any]) -> XGBClassifier:
        params = {**self.base_params, **params}
        params['tree_method'] = 'gpu_hist'  # Enable GPU
        params['predictor'] = 'gpu_predictor'
        return XGBClassifier(**params)
    
    def objective(self, trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        }
        scores, _, _ = self.fit(params)
        return np.mean(scores)

# ==========================
# 6. Model Configuration and Optimization
# ==========================

def get_model_configs(device: str, categorical_feats: List[str], num_gpus: int) -> List[Tuple[Any, dict, bool, str]]:
    """
    Define model configurations based on the device.

    Args:
        device (str): 'cpu' or 'gpu'.
        categorical_feats (List[str]): List of categorical feature names.
        num_gpus (int): Number of GPUs available.

    Returns:
        List[Tuple[Any, dict, bool, str]]: List of model classes with their configurations.
    """
    base_config = {
        'cpu': {
            'lgb': {"random_state": 42, "n_jobs": -1, "metric": "auc", 'verbosity': -1},
            'cat': {'task_type': 'CPU', 'random_seed': 42, 'eval_metric': 'AUC', 'thread_count': -1},
            'xgb': {"random_state": 42, "n_jobs": -1, "objective": "binary:logistic", "eval_metric": "auc", 'use_label_encoder': False}
        },
        'gpu': {
            'lgb': {"random_state": 42, "n_jobs": -1, "metric": "auc", 'verbosity': -1, 'device': 'gpu'},
            'cat': {'task_type': 'GPU', 'devices': ':'.join(map(str, range(num_gpus))), 'random_seed': 42, 'eval_metric': 'AUC', 'thread_count': -1},
            'xgb': {"random_state": 42, "n_jobs": -1, "objective": "binary:logistic", "eval_metric": "auc", 'use_label_encoder': False, 'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor'}
        }
    }

    return [
        (Model_goss, base_config[device]['lgb'], True, 'lightgbm_goss'),
        (Model_gbdt, base_config[device]['lgb'], True, 'lightgbm_gbdt'),
        (Model_loss, base_config[device]['cat'], True, 'catboost_lossguide'),
        (Model_sym, base_config[device]['cat'], True, 'catboost_symmetric'),
        (Model_depth, base_config[device]['cat'], True, 'catboost_depthwise'),
        (Model_gbtree, base_config[device]['xgb'], False, 'xgboost')
    ]

def model_optimization(train: pd.DataFrame, test: pd.DataFrame, target: str, categorical_feats: List[str], 
                      n_trials: int, device: str, base_path: str = 'output/') -> None:
    """
    Optimize hyperparameters for all models and save the best parameters and predictions.

    Args:
        train (pd.DataFrame): Preprocessed training DataFrame.
        test (pd.DataFrame): Preprocessed testing DataFrame.
        target (str): Target variable name.
        categorical_feats (List[str]): List of categorical feature names.
        n_trials (int): Number of Optuna trials per model.
        device (str): 'cpu' or 'gpu'.
        base_path (str, optional): Base path to save outputs. Defaults to 'output/'.
    """
    num_gpus = torch.cuda.device_count() if device == 'gpu' else 0
    print(f"Number of available GPUs: {num_gpus}")

    models = get_model_configs(device, categorical_feats, num_gpus)

    os.makedirs(os.path.join(base_path, 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'params'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'models'), exist_ok=True)

    for i, (model_cls, base_params, use_cat_feats, model_type) in enumerate(models):
        print(f"\nStarting optimization for Model {i + 1}: {model_type}")
        model_instance = model_cls(train, test, target, categorical_feats, base_params)
        best_params = model_instance.optimize(n_trials=n_trials)
        
        # Save best parameters
        joblib.dump(best_params, os.path.join(base_path, 'params', f'best_params_{i}.joblib'))
        print(f"Best parameters for {model_type} saved.\n")
        
        # Retrain model with best parameters
        scores, preds, oof_preds = model_instance.fit(best_params)
        
        # Save predictions and models
        joblib.dump(scores, os.path.join(base_path, 'predictions', f'scores_{i}.joblib'))
        joblib.dump(preds, os.path.join(base_path, 'predictions', f'preds_{i}.joblib'))
        joblib.dump(oof_preds, os.path.join(base_path, 'predictions', f'oof_preds_{i}.joblib'))
        joblib.dump(model_instance.model_dict, os.path.join(base_path, 'models', f'models_{i}.joblib'))
        
        print(f"Model {model_type} optimization and training completed.\n")

# ==========================
# 7. Blending Predictions
# ==========================

class OptunaWeights:
    """
    Optimize weights for blending model predictions using Optuna.
    """
    def __init__(self, random_state: int, n_trials: int = 5000):
        self.study: Optional[optuna.Study] = None
        self.weights: Optional[List[float]] = None
        self.random_state = random_state
        self.n_trials = n_trials
        self.selected_preds: Optional[List[bool]] = None

    def _objective(self, trial: optuna.Trial, y_true: np.ndarray, y_preds: List[np.ndarray]) -> float:
        weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))]
        selected_preds = [True for _ in range(len(y_preds))]  # Currently selecting all predictions

        selected_weights = [w for w, s in zip(weights, selected_preds) if s]
        weight_sum = sum(selected_weights)
        if weight_sum == 0:
            return 0.0
        norm_weights = [w / weight_sum for w in selected_weights]
        selected_y_preds = [pred for pred, s in zip(y_preds, selected_preds) if s]
        weighted_pred = np.average(np.array(selected_y_preds), axis=0, weights=norm_weights)
        return roc_auc_score(y_true, weighted_pred)

    def fit(self, y_true: np.ndarray, y_preds: List[np.ndarray]) -> None:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        self.study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.HyperbandPruner(),
            direction='maximize'
        )
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials, show_progress_bar=True)

        self.weights = [self.study.best_params.get(f"weight{n}", 1.0) for n in range(len(y_preds))]
        # Normalize weights
        weight_sum = sum(self.weights)
        if weight_sum == 0:
            raise ValueError("All weights are zero. Unable to normalize.")
        self.weights = [w / weight_sum for w in self.weights]

def blend_predictions(train: pd.DataFrame, test: pd.DataFrame, target: str, base_path: str = 'output/', n_trials: int = 1000) -> Tuple[np.ndarray, List[Any], np.ndarray]:
    """
    Blend model predictions using optimized weights.

    Args:
        train (pd.DataFrame): Preprocessed training DataFrame with target.
        test (pd.DataFrame): Preprocessed testing DataFrame.
        target (str): Target variable name.
        base_path (str, optional): Base path where model predictions are saved. Defaults to 'output/'.
        n_trials (int, optional): Number of Optuna trials for weight optimization. Defaults to 1000.

    Returns:
        Tuple[np.ndarray, List[Any], np.ndarray]: Final blended predictions, list of model predictions, and optimized weights.
    """
    model_preds = []
    oof_preds = []
    num_models = 6  # Adjust based on the number of models

    for i in range(num_models):
        preds_path = os.path.join(base_path, 'predictions', f'preds_{i}.joblib')
        oof_path = os.path.join(base_path, 'predictions', f'oof_preds_{i}.joblib')
        
        if not os.path.exists(preds_path) or not os.path.exists(oof_path):
            raise FileNotFoundError(f"Predictions for model {i} not found.")
        
        preds = joblib.load(preds_path)
        oof = joblib.load(oof_path)
        model_preds.append(preds)
        oof_preds.append(oof)
        print(f"Loaded predictions for Model {i + 1}")
    
    # Initialize OptunaWeights
    ow = OptunaWeights(random_state=42, n_trials=n_trials)
    ow.fit(train[target].values, y_preds=oof_preds)

    # Apply weights to test predictions
    final_pred = np.zeros(len(test))
    for weight, pred in zip(ow.weights, model_preds):
        final_pred += weight * pred

    print(f"Blending completed with weights: {ow.weights}")
    return final_pred, ow.weights, model_preds

# ==========================
# 8. Generate Submission
# ==========================

def generate_submission(final_pred: np.ndarray, sample_submission_path: str, output_path: str = 'submission.csv') -> None:
    """
    Create the submission CSV file.

    Args:
        final_pred (np.ndarray): Final blended predictions.
        sample_submission_path (str): Path to the sample submission CSV.
        output_path (str, optional): Path to save the submission CSV. Defaults to 'submission.csv'.
    """
    submission = pd.read_csv(sample_submission_path)
    if 'loan_status' in submission.columns:
        submission['loan_status'] = final_pred
    elif 'Loan_Status' in submission.columns:
        submission['Loan_Status'] = final_pred
    else:
        raise ValueError("The sample submission file must contain 'loan_status' or 'Loan_Status' column.")
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

# ==========================
# 9. Main Execution
# ==========================

def main():
    # Load data
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
    
    # Preprocess data
    train_preprocessed, test_preprocessed = preprocess_data(train_df, test_df, TARGET)
    
    # Identify categorical features
    categorical_features = train_preprocessed.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical features: {categorical_features}")
    
    # Determine device
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Optimize models
    model_optimization(
        train=train_preprocessed,
        test=test_preprocessed,
        target=TARGET,
        categorical_feats=categorical_features,
        n_trials=50,  # Adjust based on computational resources
        device=device,
        base_path=OUTPUT_DIR
    )
    
    # Blend predictions
    final_pred, final_weights, model_preds = blend_predictions(
        train=train_preprocessed,
        test=test_preprocessed,
        target=TARGET,
        base_path=OUTPUT_DIR,
        n_trials=500  # Adjust based on computational resources
    )
    
    # Load sample submission
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    
    # Ensure the order of predictions matches the submission
    if len(final_pred) != len(submission_df):
        raise ValueError("The number of predictions does not match the number of entries in the submission file.")
    
    # Assign predictions to submission
    if 'loan_status' in submission_df.columns:
        submission_df['loan_status'] = final_pred
    elif 'Loan_Status' in submission_df.columns:
        submission_df['Loan_Status'] = final_pred
    else:
        raise ValueError("The sample submission file must contain 'loan_status' or 'Loan_Status' column.")
    
    # Save submission
    submission_df.to_csv('/kaggle/working/submission.csv', index=False)
    print("Submission file 'submission.csv' created successfully!")

if __name__ == "__main__":
    main()
