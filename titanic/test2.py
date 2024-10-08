#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:23:16 2024

@author: youknowjp
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def predict_titanic_survival(train_path, test_path):
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
    
    # Convert categorical variables into numerical ones
    label_encoder = LabelEncoder()
    train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
    test_data['Sex'] = label_encoder.transform(test_data['Sex'])
    
    train_data['Embarked'] = label_encoder.fit_transform(train_data['Embarked'])
    test_data['Embarked'] = label_encoder.transform(test_data['Embarked'])
    
    # Select relevant features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']
    X = train_data[features]
    y = train_data['Survived']
    
    # Split the training data for evaluation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # List of models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVC': SVC(random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        results[name] = val_accuracy
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values(), color='skyblue')
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title('Model Comparison')
    plt.xticks(rotation=45)
    plt.show()
    
    return results

# Paths to the datasets
train_path = '/Users/youknowjp/Downloads/titanic/train.csv'
test_path = '/Users/youknowjp/Downloads/titanic/test.csv'

# Call the function and get the accuracy
results = predict_titanic_survival(train_path, test_path)

print("Model Accuracy Scores:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")
