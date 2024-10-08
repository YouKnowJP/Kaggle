#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 00:20:56 2024

@author: youknowjp
"""

# Import necessary libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

# Ensure you have the necessary NLTK data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset (train.csv)
train_df = pd.read_csv('/Users/youknowjp/Downloads/nlp-getting-started/train.csv')

# Function to clean the text
def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Apply the cleaning function to the 'text' column
train_df['cleaned_text'] = train_df['text'].apply(clean_text)

# Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

train_df['sentiment'] = train_df['cleaned_text'].apply(get_sentiment)

# Feature Engineering: Create new features
train_df['tweet_length'] = train_df['cleaned_text'].apply(len)
train_df['word_count'] = train_df['cleaned_text'].apply(lambda x: len(x.split()))
train_df['hashtag_count'] = train_df['text'].apply(lambda x: len(re.findall(r'\#\w+', x)))

# TF-IDF Vectorization with Trigrams (limiting to 5000 features)
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2, max_df=0.9)
tfidf_features = tfidf_vectorizer.fit_transform(train_df['cleaned_text'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Combine TF-IDF with additional length-based features
length_features_df = train_df[['tweet_length', 'word_count', 'hashtag_count', 'sentiment']].reset_index(drop=True)
combined_features_df = pd.concat([tfidf_df, length_features_df], axis=1)

# Encode the 'keyword' column (One-hot encoding)
train_df['keyword'].fillna('missing', inplace=True)
keyword_encoded_df = pd.get_dummies(train_df['keyword'], prefix='keyword')

# Combine the keyword features with the TF-IDF and length-based features
final_combined_features = pd.concat([combined_features_df, keyword_encoded_df], axis=1)

# Define target variable
X = final_combined_features
y = train_df['target']

# Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear SVM model with class weighting
svm_model = LinearSVC(class_weight='balanced', max_iter=10000)

# Hyperparameter tuning with GridSearchCV
param_grid = {'C': [0.01, 0.1, 1, 10]}  # Regularization parameter
grid_search = GridSearchCV(svm_model, param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Best model from Grid Search
best_svm_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_svm_model.predict(X_test)

# Evaluate the model using F1 score and classification report
f1 = f1_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Output the evaluation results
print(f"F1 Score: {f1:.4f}")
print("Classification Report:\n", classification_rep)
