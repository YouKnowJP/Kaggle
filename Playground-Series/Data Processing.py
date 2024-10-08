#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 17:30:03 2024

@author: youknowjp
"""

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def load_and_preprocess_data(train_path, test_path):
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Check for missing values
    print("=== Train Data Missing Values ===")
    print(train_data.isnull().sum())
    print("\n=== Test Data Missing Values ===")
    print(test_data.isnull().sum())

    # Get the data types of each column
    print("\n=== Train Data Types ===")
    print(train_data.dtypes)
    print("\n=== Test Data Types ===")
    print(test_data.dtypes)
    
    return train_data, test_data

def visualize_data(train_data):
    # Distribution of target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Target', data=train_data)
    plt.title('Distribution of Target Variable')
    plt.show()

    # Correlation matrix
    plt.figure(figsize=(14, 12))
    correlation_matrix = train_data.corr(numeric_only=True)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Identify highly correlated features
    threshold = 0.8
    highly_correlated_features = [
        (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
        for i in range(len(correlation_matrix.columns))
        for j in range(i)
        if abs(correlation_matrix.iloc[i, j]) > threshold
    ]

    # Display the highly correlated features
    highly_correlated_df = pd.DataFrame(highly_correlated_features, columns=['Feature 1', 'Feature 2', 'Correlation'])
    print("=== Highly Correlated Features (Correlation > 0.8) ===")
    print(highly_correlated_df)

def encode_and_split_data(train_data):
    # Encode the 'Target' variable
    label_encoder = LabelEncoder()
    train_data['Target_encoded'] = label_encoder.fit_transform(train_data['Target'])

    # Define features and target
    features = train_data.drop(['id', 'Target', 'Target_encoded'], axis=1)
    target = train_data['Target_encoded']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Initialize the model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of XGBoost: {accuracy:.4f}")

# Main execution
train_path = "/Users/youknowjp/Downloads/playground-series-s4e6/train.csv"
test_path = "/Users/youknowjp/Downloads/playground-series-s4e6/test.csv"

train_data, test_data = load_and_preprocess_data(train_path, test_path)
visualize_data(train_data)
X_train_scaled, X_test_scaled, y_train, y_test = encode_and_split_data(train_data)
train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
