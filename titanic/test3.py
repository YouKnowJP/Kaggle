#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:23:16 2024

@author: youknowjp
"""

import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def predict_titanic_survival(train_path, test_path, submission_path):
    # Load the datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Handle missing values for 'Age', 'Fare', and 'Embarked'
    imputer = SimpleImputer(strategy='median')
    train_data['Age'] = imputer.fit_transform(train_data[['Age']])
    test_data['Age'] = imputer.transform(test_data[['Age']])
    
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
    train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
    test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)
    
    # Feature engineering: Create new features
    train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
    test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
    
    train_data['IsAlone'] = 1
    train_data['IsAlone'].loc[train_data['FamilySize'] > 1] = 0
    test_data['IsAlone'] = 1
    test_data['IsAlone'].loc[test_data['FamilySize'] > 1] = 0
    
    # Convert categorical variables into numerical ones
    label_encoder = LabelEncoder()
    train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
    test_data['Sex'] = label_encoder.transform(test_data['Sex'])
    
    train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
    test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])
    
    # Select relevant features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone']
    X = train_data[features]
    y = train_data['Survived']
    X_test = test_data[features]
    
    # Split the training data for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Define the objective function for Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 1000)
        max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        rf_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=-1
        )
        score = cross_val_score(rf_model, X_train, y_train, cv=StratifiedKFold(n_splits=5), scoring='accuracy').mean()
        return score
    
    # Optimize hyperparameters
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    
    best_params = study.best_params
    best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    best_model.fit(X_train, y_train)
    
    # Make predictions on the validation set
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    
    # Make predictions on the test set
    y_test_pred = best_model.predict(X_test)
    
    # Save the submission file
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': y_test_pred
    })
    submission.to_csv(submission_path, index=False)
    
    print(f"Best parameters found: {best_params}")
    print(f"Validation Accuracy: {val_accuracy}")
    
    return val_accuracy

# Paths to the datasets
train_path = '/Users/youknowjp/Downloads/titanic/train.csv'
test_path = '/Users/youknowjp/Downloads/titanic/test.csv'
submission_path = '/Users/youknowjp/Downloads/titanic/submission.csv'

# Call the function and get the accuracy
val_accuracy = predict_titanic_survival(train_path, test_path, submission_path)
