#!/usr/bin/env python3
"""
Simplified training script for tweet disaster classification
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
except:
    pass

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#\w+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)

def main():
    print("🚀 Starting Tweet Disaster Classification Training")
    print("="*60)
    
    # Load data
    print("\n1️⃣ Loading data...")
    df = pd.read_csv('data/twitter_disaster.csv')
    print(f"✓ Data loaded: {df.shape[0]} samples")
    
    # Preprocess
    print("\n2️⃣ Preprocessing text...")
    preprocessor = TextPreprocessor()
    df['cleaned_text'] = df['text'].apply(preprocessor.clean_text)
    print(f"✓ Text preprocessed")
    
    # Prepare data
    print("\n3️⃣ Preparing data...")
    X = df['cleaned_text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Feature extraction
    print("\n4️⃣ Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"✓ Features extracted: {X_train_tfidf.shape[1]} features")
    
    # Train and compare models
    print("\n5️⃣ Training models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'SVM': SVC(kernel='linear', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'model': name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
        
        print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        # Save each model
        joblib.dump(model, f'model/{name.replace(" ", "_").lower()}.pkl')
        joblib.dump(vectorizer, f'model/{name.replace(" ", "_").lower()}_vectorizer.pkl')
    
    # Hyperparameter tuning for best model
    print("\n6️⃣ Hyperparameter tuning...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    param_grid = {
        'tfidf__max_features': [3000, 5000],
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear']
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"✓ Best parameters: {grid_search.best_params_}")
    print(f"✓ Best F1 score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    y_pred = best_model.predict(X_test)
    print(f"\n📊 Best Model Performance:")
    print(classification_report(y_test, y_pred, target_names=['Non-Disaster', 'Disaster']))
    
    # Save best model
    joblib.dump(best_model, 'model/best_logistic_regression.pkl')
    print("✓ Best model saved to model/best_logistic_regression.pkl")
    
    # Save comparison results
    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('f1_score', ascending=False)
    comparison_df.to_csv('model/comparison_results.csv', index=False)
    print("✓ Comparison results saved to model/comparison_results.csv")
    
    # Test predictions
    print("\n7️⃣ Testing predictions...")
    test_tweets = [
        "Earthquake hits California, massive damage reported",
        "Just had a great lunch with friends",
        "Forest fire spreading rapidly, evacuate immediately",
        "The weather is nice today",
        "Tornado warning issued for Oklahoma"
    ]
    
    for tweet in test_tweets:
        cleaned = preprocessor.clean_text(tweet)
        prediction = best_model.predict([cleaned])[0]
        probability = best_model.predict_proba([cleaned])[0][1]
        label = 'Disaster' if prediction == 1 else 'Non-Disaster'
        print(f"\n  Tweet: '{tweet}'")
        print(f"  Prediction: {label} ({probability:.4f})")
    
    print("\n✅ Training completed successfully!")
    print("\n📁 Files created:")
    print("   - model/best_logistic_regression.pkl")
    print("   - model/comparison_results.csv")
    print("   - Individual model files")

if __name__ == "__main__":
    main()