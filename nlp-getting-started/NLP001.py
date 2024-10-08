#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:09:45 2024

@author: youknowjp
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Load the datasets
train_data = pd.read_csv('/Users/youknowjp/Downloads/nlp-getting-started/train.csv')
test_data = pd.read_csv('/Users/youknowjp/Downloads/nlp-getting-started/test.csv')

# Handle missing values
train_data['keyword'].fillna('', inplace=True)
train_data['location'].fillna('', inplace=True)
test_data['keyword'].fillna('', inplace=True)
test_data['location'].fillna('', inplace=True)

# Create 'combined_text' feature by concatenating keyword, location, and text
train_data['combined_text'] = train_data['keyword'] + ' ' + train_data['location'] + ' ' + train_data['text']
test_data['combined_text'] = test_data['keyword'] + ' ' + test_data['location'] + ' ' + test_data['text']

# Add 'text_length' feature as the number of words in the tweet
train_data['text_length'] = train_data['text'].apply(lambda x: len(x.split()))
test_data['text_length'] = test_data['text'].apply(lambda x: len(x.split()))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_data[['combined_text', 'text_length']], 
                                                  train_data['target'], 
                                                  test_size=0.2, 
                                                  random_state=42)

# Define the preprocessor for text (TF-IDF) and numerical (text_length) features
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english'), 'combined_text'),
        ('length', StandardScaler(), ['text_length'])
    ])

# Create a pipeline with the preprocessor and XGBoost classifier
enhanced_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

# Define the parameter grid for Randomized Search
param_dist = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [3, 5, 7],
    'clf__learning_rate': [0.01, 0.1, 0.3],
    'clf__subsample': [0.7, 0.8, 1.0],
    'clf__colsample_bytree': [0.7, 0.8, 1.0],
}

# Setup the RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=enhanced_pipeline, 
                                   param_distributions=param_dist,
                                   n_iter=20, 
                                   cv=3, 
                                   verbose=2, 
                                   random_state=42, 
                                   n_jobs=-1)

# Perform the search
random_search.fit(X_train, y_train)

# Get the best parameters
best_params = random_search.best_params_
print("Best Parameters:", best_params)

# Get the best estimator and evaluate it
best_model = random_search.best_estimator_

# Predict on validation set
y_pred = best_model.predict(X_val)

# Generate a classification report
print(classification_report(y_val, y_pred))
