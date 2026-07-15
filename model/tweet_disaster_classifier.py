#!/usr/bin/env python3
"""
Tweet Disaster Classification Model
====================================

A machine learning model to classify tweets as disaster-related or not.
Uses NLP techniques with TF-IDF and various classifiers.

Author: Bijender
Project: Tweet Disaster Classification
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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
except:
    pass

class TweetDisasterClassifier:
    """
    A comprehensive tweet disaster classification model with preprocessing,
    feature extraction, and multiple classifier options.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the classifier.
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.model = None
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self, file_path=None):
        """
        Load the dataset from CSV file.
        
        Args:
            file_path (str): Path to CSV file. If None, uses self.data_path
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        path = file_path or self.data_path
        if path:
            self.data = pd.read_csv(path)
            print(f"✓ Data loaded from {path}")
            print(f"  Shape: {self.data.shape}")
            print(f"  Columns: {list(self.data.columns)}")
            return self.data
        else:
            raise ValueError("No data path provided")
    
    def explore_data(self):
        """
        Perform exploratory data analysis on the dataset.
        """
        if self.data is None:
            print("Please load data first using load_data()")
            return
            
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic info
        print("\n📊 Dataset Info:")
        print(self.data.info())
        
        # Target distribution
        print("\n🎯 Target Distribution:")
        target_counts = self.data['target'].value_counts()
        print(target_counts)
        
        # Plot target distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='target', data=self.data)
        plt.title('Target Distribution (0: Non-Disaster, 1: Disaster)')
        plt.xlabel('Target')
        plt.ylabel('Count')
        plt.savefig('model/plots/target_distribution.png')
        plt.close()
        
        # Text length analysis
        self.data['text_length'] = self.data['text'].apply(len)
        print("\n📏 Text Length Statistics:")
        print(self.data['text_length'].describe())
        
        # Plot text length by target
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='target', y='text_length', data=self.data)
        plt.title('Text Length Distribution by Target')
        plt.xlabel('Target')
        plt.ylabel('Text Length')
        plt.savefig('model/plots/text_length_distribution.png')
        plt.close()
        
        # Sample tweets
        print("\n🐦 Sample Disaster Tweets:")
        disaster_samples = self.data[self.data['target'] == 1]['text'].head(5)
        for i, tweet in enumerate(disaster_samples, 1):
            print(f"  {i}. {tweet}")
            
        print("\n🐦 Sample Non-Disaster Tweets:")
        non_disaster_samples = self.data[self.data['target'] == 0]['text'].head(5)
        for i, tweet in enumerate(non_disaster_samples, 1):
            print(f"  {i}. {tweet}")
    
    def preprocess_text(self, text):
        """
        Preprocess tweet text for NLP tasks.
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned and preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'\@\w+|\#\w+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        # Join tokens back to string
        cleaned_text = ' '.join(tokens)
        
        return cleaned_text
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare the data for training and testing.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            print("Please load data first using load_data()")
            return
            
        # Clean text
        print("🧹 Cleaning text data...")
        self.data['cleaned_text'] = self.data['text'].apply(self.preprocess_text)
        
        # Encode labels
        self.data['target_encoded'] = self.label_encoder.fit_transform(self.data['target'])
        
        # Split data
        X = self.data['cleaned_text']
        y = self.data['target_encoded']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"✓ Data prepared for training")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Test samples: {len(self.X_test)}")
        print(f"  Classes: {self.label_encoder.classes_}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def extract_features(self, vectorizer_type='tfidf', max_features=5000):
        """
        Extract features from text using vectorization.
        
        Args:
            vectorizer_type (str): 'tfidf' or 'count'
            max_features (int): Maximum number of features to keep
            
        Returns:
            sklearn vectorizer: Fitted vectorizer
        """
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.7
            )
        elif vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.7
            )
        else:
            raise ValueError("vectorizer_type must be 'tfidf' or 'count'")
            
        # Fit and transform training data
        X_train_features = self.vectorizer.fit_transform(self.X_train)
        X_test_features = self.vectorizer.transform(self.X_test)
        
        self.X_train_features = X_train_features
        self.X_test_features = X_test_features
        
        print(f"✓ Features extracted using {vectorizer_type.upper()}")
        print(f"  Feature shape: {X_train_features.shape}")
        
        return self.vectorizer
    
    def train_model(self, model_name='logistic_regression', **kwargs):
        """
        Train a classification model.
        
        Args:
            model_name (str): Name of the model to train
            **kwargs: Additional arguments for the model
            
        Returns:
            sklearn model: Trained model
        """
        models = {
            'logistic_regression': LogisticRegression,
            'naive_bayes': MultinomialNB,
            'svm': SVC,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'knn': KNeighborsClassifier
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
            
        print(f"🎓 Training {model_name.replace('_', ' ').title()}...")
        
        # Get model class
        ModelClass = models[model_name]
        
        # Set default parameters based on model
        if model_name == 'logistic_regression':
            default_params = {'max_iter': 1000, 'random_state': 42}
        elif model_name == 'naive_bayes':
            default_params = {'alpha': 1.0}
        elif model_name == 'svm':
            default_params = {'kernel': 'linear', 'random_state': 42}
        elif model_name == 'random_forest':
            default_params = {'n_estimators': 100, 'random_state': 42}
        elif model_name == 'gradient_boosting':
            default_params = {'n_estimators': 100, 'random_state': 42}
        elif model_name == 'knn':
            default_params = {'n_neighbors': 5}
        else:
            default_params = {}
            
        # Merge with user-provided kwargs
        params = {**default_params, **kwargs}
        
        # Initialize and train model
        self.model = ModelClass(**params)
        self.model.fit(self.X_train_features, self.y_train)
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_features)
        y_pred_proba = self.model.predict_proba(self.X_test_features)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Evaluate
        self.evaluate_model(y_pred, y_pred_proba)
        
        return self.model
    
    def evaluate_model(self, y_pred, y_pred_proba=None):
        """
        Evaluate the trained model.
        
        Args:
            y_pred (array): Predicted labels
            y_pred_proba (array): Predicted probabilities (optional)
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print(f"📊 Accuracy:  {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall:    {recall:.4f}")
        print(f"📊 F1-Score:  {f1:.4f}")
        
        if y_pred_proba is not None:
            try:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                print(f"📊 ROC AUC:   {roc_auc:.4f}")
            except:
                pass
                
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Non-Disaster', 'Disaster']))
        
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Disaster', 'Disaster'],
                   yticklabels=['Non-Disaster', 'Disaster'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('model/plots/confusion_matrix.png')
        plt.close()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def compare_models(self, models_to_compare=None):
        """
        Compare multiple models on the same dataset.
        
        Args:
            models_to_compare (list): List of model names to compare
            
        Returns:
            pd.DataFrame: Comparison results
        """
        if models_to_compare is None:
            models_to_compare = ['logistic_regression', 'naive_bayes', 'svm', 'random_forest']
            
        print(f"\n🔬 Comparing {len(models_to_compare)} models...")
        
        results = []
        
        for model_name in models_to_compare:
            print(f"\nTraining {model_name}...")
            try:
                model = self.train_model(model_name)
                y_pred = model.predict(self.X_test_features)
                y_pred_proba = model.predict_proba(self.X_test_features)[:, 1] if hasattr(model, 'predict_proba') else None
                
                metrics = self.evaluate_model(y_pred, y_pred_proba)
                metrics['model'] = model_name
                results.append(metrics)
                
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                
        # Create comparison dataframe
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1_score']]
        comparison_df = comparison_df.sort_values('f1_score', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON RESULTS")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('model/comparison_results.csv', index=False)
        
        return comparison_df
    
    def hyperparameter_tuning(self, model_name='logistic_regression', param_grid=None):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name (str): Name of the model to tune
            param_grid (dict): Parameter grid for GridSearchCV
            
        Returns:
            GridSearchCV: Fitted grid search object
        """
        if param_grid is None:
            # Default parameter grids
            param_grids = {
                'logistic_regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'naive_bayes': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                },
                'svm': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20]
                }
            }
            param_grid = param_grids.get(model_name, {})
            
        if not param_grid:
            print("No parameter grid provided and no default available for this model")
            return None
            
        print(f"🎯 Performing hyperparameter tuning for {model_name}...")
        
        # Get base model
        models = {
            'logistic_regression': LogisticRegression,
            'naive_bayes': MultinomialNB,
            'svm': SVC,
            'random_forest': RandomForestClassifier
        }
        
        ModelClass = models.get(model_name)
        if ModelClass is None:
            raise ValueError(f"Unknown model: {model_name}")
            
        # Create pipeline
        pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', ModelClass())
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid={f'classifier__{k}': v for k, v in param_grid.items()},
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit on training data (need to use raw text)
        grid_search.fit(self.X_train, self.y_train)
        
        # Save best model
        self.best_model = grid_search.best_estimator_
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best F1 score: {grid_search.best_score_:.4f}")
        
        # Evaluate best model
        y_pred = self.best_model.predict(self.X_test)
        self.evaluate_model(y_pred)
        
        return grid_search
    
    def save_model(self, model=None, file_path='model/tweet_disaster_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            model: Model to save (if None, uses self.model)
            file_path (str): Path to save the model
        """
        model_to_save = model or self.model or self.best_model
        if model_to_save is None:
            print("No model to save. Train a model first.")
            return
            
        joblib.dump(model_to_save, file_path)
        print(f"✓ Model saved to {file_path}")
        
        return file_path
    
    def load_saved_model(self, file_path='model/tweet_disaster_model.pkl'):
        """
        Load a saved model from disk.
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(file_path)
        self.model = model
        print(f"✓ Model loaded from {file_path}")
        return model
    
    def predict(self, text, model=None):
        """
        Make a prediction on new text.
        
        Args:
            text (str or list): Text to classify
            model: Model to use (if None, uses self.model or self.best_model)
            
        Returns:
            Prediction result
        """
        model_to_use = model or self.model or self.best_model
        if model_to_use is None:
            print("No model available. Train or load a model first.")
            return None
            
        # Preprocess text
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
            
        cleaned_texts = [self.preprocess_text(t) for t in texts]
        
        # Vectorize
        if hasattr(model_to_use, 'named_steps'):
            # Pipeline model
            features = model_to_use.named_steps['vectorizer'].transform(cleaned_texts)
            predictions = model_to_use.named_steps['classifier'].predict(features)
            probabilities = model_to_use.named_steps['classifier'].predict_proba(features)
        else:
            # Regular model
            if self.vectorizer is None:
                print("Vectorizer not available. Extract features first.")
                return None
            features = self.vectorizer.transform(cleaned_texts)
            predictions = model_to_use.predict(features)
            probabilities = model_to_use.predict_proba(features) if hasattr(model_to_use, 'predict_proba') else None
            
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Prepare results
        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predicted_labels, probabilities or [None]*len(predictions))):
            result = {
                'text': text,
                'prediction': 'Disaster' if pred == 1 else 'Non-Disaster',
                'confidence': prob[1] if prob is not None else None
            }
            results.append(result)
            
        if len(results) == 1:
            return results[0]
        else:
            return results
    
    def create_visualizations(self):
        """
        Create various visualizations for the model and data.
        """
        import os
        os.makedirs('model/plots', exist_ok=True)
        
        # Target distribution (already created in explore_data)
        # Text length distribution (already created in explore_data)
        
        # Word cloud for disaster tweets
        try:
            from wordcloud import WordCloud
            
            disaster_text = ' '.join(self.data[self.data['target'] == 1]['cleaned_text'])
            non_disaster_text = ' '.join(self.data[self.data['target'] == 0]['cleaned_text'])
            
            # Disaster word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(disaster_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Disaster Tweets - Word Cloud')
            plt.axis('off')
            plt.savefig('model/plots/disaster_wordcloud.png')
            plt.close()
            
            # Non-disaster word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(non_disaster_text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Non-Disaster Tweets - Word Cloud')
            plt.axis('off')
            plt.savefig('model/plots/non_disaster_wordcloud.png')
            plt.close()
            
        except ImportError:
            print("WordCloud not available. Install with: pip install wordcloud")
            
        print("✓ Visualizations created in model/plots/")


def main():
    """
    Main function to run the complete pipeline.
    """
    print("🚀 Starting Tweet Disaster Classification Model")
    print("="*60)
    
    # Initialize classifier
    classifier = TweetDisasterClassifier(data_path='data/twitter_disaster.csv')
    
    # Step 1: Load data
    print("\n1️⃣ Loading data...")
    classifier.load_data()
    
    # Step 2: Explore data
    print("\n2️⃣ Exploring data...")
    classifier.explore_data()
    
    # Step 3: Prepare data
    print("\n3️⃣ Preparing data...")
    classifier.prepare_data()
    
    # Step 4: Extract features
    print("\n4️⃣ Extracting features...")
    classifier.extract_features(vectorizer_type='tfidf', max_features=5000)
    
    # Step 5: Compare models
    print("\n5️⃣ Comparing models...")
    comparison_results = classifier.compare_models([
        'logistic_regression',
        'naive_bayes',
        'svm',
        'random_forest'
    ])
    
    # Step 6: Hyperparameter tuning for best model
    print("\n6️⃣ Performing hyperparameter tuning...")
    classifier.hyperparameter_tuning(
        model_name='logistic_regression',
        param_grid={
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
    )
    
    # Step 7: Save best model
    print("\n7️⃣ Saving model...")
    classifier.save_model(file_path='model/tweet_disaster_model.pkl')
    
    # Step 8: Create visualizations
    print("\n8️⃣ Creating visualizations...")
    classifier.create_visualizations()
    
    # Step 9: Test predictions
    print("\n9️⃣ Testing predictions...")
    test_tweets = [
        "Earthquake hits California, massive damage reported",
        "Just had a great lunch with friends",
        "Forest fire spreading rapidly, evacuate immediately",
        "The weather is nice today",
        "Tornado warning issued for Oklahoma"
    ]
    
    for tweet in test_tweets:
        result = classifier.predict(tweet)
        print(f"\n📝 Tweet: '{tweet}'")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.4f}")
    
    print("\n✅ Pipeline completed successfully!")
    print("\n📁 Files created:")
    print("   - model/tweet_disaster_model.pkl (trained model)")
    print("   - model/comparison_results.csv (model comparison)")
    print("   - model/plots/ (visualizations)")


if __name__ == "__main__":
    main()