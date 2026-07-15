#!/usr/bin/env python3
"""
Simple prediction script for tweet disaster classification

Usage:
    python predict.py "Your tweet text here"
    
    or
    
    python predict.py  # Interactive mode
"""

import sys
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
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
    print("🚨 Tweet Disaster Classifier")
    print("="*50)
    
    # Load model
    try:
        model = joblib.load('model/best_logistic_regression.pkl')
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first.")
        print("   Run: python train_model.py")
        return
    
    preprocessor = TextPreprocessor()
    
    # Check if tweet provided as command line argument
    if len(sys.argv) > 1:
        tweet = ' '.join(sys.argv[1:])
        predict_tweet(model, preprocessor, tweet)
    else:
        # Interactive mode
        print("\n📝 Enter tweets to classify (type 'quit' to exit):")
        print("-"*50)
        
        while True:
            try:
                tweet = input("\nEnter tweet: ").strip()
                if tweet.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                if tweet:
                    predict_tweet(model, preprocessor, tweet)
                else:
                    print("Please enter a tweet.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

def predict_tweet(model, preprocessor, tweet):
    """Predict and display result for a single tweet."""
    # Preprocess
    cleaned_text = preprocessor.clean_text(tweet)
    
    # Predict
    prediction = model.predict([cleaned_text])[0]
    probability = model.predict_proba([cleaned_text])[0][1]
    
    # Display result
    label = '🔴 Disaster' if prediction == 1 else '🟢 Non-Disaster'
    confidence = probability if prediction == 1 else 1 - probability
    
    print(f"\n📝 Tweet: {tweet}")
    print(f"🎯 Prediction: {label}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"📈 Disaster Probability: {probability:.2%}")

if __name__ == "__main__":
    main()