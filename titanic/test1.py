#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 23:40:34 2024

@author: youknowjp
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def predict_titanic_survival(train_path, test_path):
    # Load the datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    train_data['Age'] = imputer.fit_transform(train_data[['Age']])
    test_data['Age'] = imputer.transform(test_data[['Age']])

    train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
    test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

    # Convert categorical variables into numerical ones
    label_encoder = LabelEncoder()
    train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
    test_data['Sex'] = label_encoder.transform(test_data['Sex'])

    train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
    test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])

    # Select relevant features
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = train_data[features]
    y = train_data['Survived']

    # Split the training data for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the validation set
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    return val_accuracy

# Paths to the datasets
train_path = '/Users/youknowjp/Downloads/titanic/train.csv'
test_path = '/Users/youknowjp/Downloads/titanic/test.csv'

# Call the function and get the accuracy
val_accuracy = predict_titanic_survival(train_path, test_path)

print(f"Validation Accuracy: {val_accuracy}")
