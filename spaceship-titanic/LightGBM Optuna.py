#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:29:57 2024

@author: youknowjp
"""

import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
# 
# Load the datasets
train_data = pd.read_csv('/Users/youknowjp/Downloads/spaceship-titanic/train.csv')

# Feature engineering: Extract deck, num, and side from the cabin
def process_cabin(data):
    data['Deck'] = data['Cabin'].str[0]
    data['Num'] = data['Cabin'].str.extract('(\d+)').astype(float)
    data['Side'] = data['Cabin'].str[-1]
    data = data.drop(columns=['Cabin', 'Name'])
    return data

# Feature engineering: Additional features
def additional_features(data):
    # Handle missing values before creating new features
    data['RoomService'] = data['RoomService'].fillna(0)
    data['FoodCourt'] = data['FoodCourt'].fillna(0)
    data['ShoppingMall'] = data['ShoppingMall'].fillna(0)
    data['Spa'] = data['Spa'].fillna(0)
    data['VRDeck'] = data['VRDeck'].fillna(0)
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    data['VIP'] = data['VIP'].fillna(False).astype(int)
    
    # Family size
    data['FamilySize'] = data['Num'].groupby(data['PassengerId'].str.split('_').str[0]).transform('count')
    
    # Is alone
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Total spend
    data['TotalSpend'] = data[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    
    # Age group
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 50, 80], labels=['Child', 'Teenager', 'Adult', 'Senior'])
    
    # Interaction features
    data['RoomService_Age'] = data['RoomService'] * data['Age']
    data['FoodCourt_Age'] = data['FoodCourt'] * data['Age']
    data['ShoppingMall_Age'] = data['ShoppingMall'] * data['Age']
    data['Spa_Age'] = data['Spa'] * data['Age']
    data['VRDeck_Age'] = data['VRDeck'] * data['Age']
    data['RoomService_VIP'] = data['RoomService'] * data['VIP']
    data['TotalSpend_Age'] = data['TotalSpend'] * data['Age']
    data['TotalSpend_VIP'] = data['TotalSpend'] * data['VIP']
    
    return data

train_data = process_cabin(train_data)
train_data = additional_features(train_data)

# Split the data into features and target
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']

# Preprocessing pipelines for numerical and categorical features
numerical_features = X.select_dtypes(include=['float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numerical_pipeline = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Define the objective function for Optuna
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0)
    }
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**param, random_state=42))
    ])
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# Run Optuna optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Return the best hyperparameters
best_params
