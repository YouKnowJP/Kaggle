#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:09:50 2024

@author: youknowjp
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

# Load the datasets
train_data = pd.read_csv('/Users/youknowjp/Downloads/spaceship-titanic/train.csv')
test_data = pd.read_csv('/Users/youknowjp/Downloads/spaceship-titanic/test.csv')

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
    data['FamilySize'] = data.groupby(data['PassengerId'].str.split('_').str[0])['Num'].transform('count')
    
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

# Apply feature engineering to both train and test data
train_data = process_cabin(train_data)
test_data = process_cabin(test_data)

train_data = additional_features(train_data)
test_data = additional_features(test_data)

# Ensure the same columns in test data as in train data
missing_cols = set(train_data.columns) - set(test_data.columns)
for c in missing_cols:
    test_data[c] = 0
test_data = test_data[train_data.columns.drop('Transported')]

# Split the data into features and target
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']

# Preprocessing pipelines for numerical and categorical features
numerical_features = X.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

# Best hyperparameters found by Optuna
best_params = {
    'n_estimators': 677,
    'max_depth': 7,
    'learning_rate': 0.010828875637296762,
    'subsample': 0.8372210952739336,
    'colsample_bytree': 0.949467714478127
}

# Define the model with the best hyperparameters
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(**best_params, random_state=42))
])

# Split the training data for model evaluation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_valid, y_valid)
print(f"Validation Accuracy: {score}")

# Preprocess the test data
X_test_preprocessed = preprocessor.transform(test_data)

# Check for any remaining NaN values and handle them
if pd.isnull(X_test_preprocessed).sum().sum() > 0:
    print("Handling remaining NaN values in test data.")
    X_test_preprocessed = SimpleImputer(strategy='mean').fit_transform(X_test_preprocessed)

# Predict on the test dataset
predictions = model.predict(X_test_preprocessed)

# Convert predictions to True/False
predictions = [bool(pred) for pred in predictions]

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions
})

submission.to_csv('/Users/youknowjp/Downloads/spaceship-titanic/submission_lgb_optuna.csv', index=False)

# Display the first few rows of the submission
print(submission.head())
