#!/usr/bin/env python3
"""
Tweet Disaster Classification Web Application
==============================================

A simple Flask web application for classifying tweets as disaster-related or not.

Usage:
    python app.py

Then open your browser to: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
except:
    pass

class TextPreprocessor:
    """Class for text preprocessing operations."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text for NLP tasks."""
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

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Load the trained model
try:
    model = joblib.load('model/best_logistic_regression.pkl')
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found. Please train the model first.")
    print("   Run: python model/tweet_disaster_classifier.py")
    model = None

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make a prediction on the submitted tweet."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get tweet from request
    data = request.get_json()
    tweet = data.get('tweet', '')
    
    if not tweet:
        return jsonify({'error': 'No tweet provided'}), 400
    
    # Preprocess and predict
    cleaned_text = preprocessor.clean_text(tweet)
    prediction = model.predict([cleaned_text])[0]
    probability = model.predict_proba([cleaned_text])[0][1]
    
    # Prepare result
    result = {
        'tweet': tweet,
        'prediction': 'Disaster' if prediction == 1 else 'Non-Disaster',
        'probability': float(probability),
        'confidence': float(probability) if prediction == 1 else float(1 - probability)
    }
    
    return jsonify(result)

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Make predictions on multiple tweets."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Get tweets from request
    data = request.get_json()
    tweets = data.get('tweets', [])
    
    if not tweets:
        return jsonify({'error': 'No tweets provided'}), 400
    
    # Preprocess and predict
    cleaned_texts = [preprocessor.clean_text(tweet) for tweet in tweets]
    predictions = model.predict(cleaned_texts)
    probabilities = model.predict_proba(cleaned_texts)
    
    # Prepare results
    results = []
    for i, (tweet, pred, prob) in enumerate(zip(tweets, predictions, probabilities)):
        result = {
            'tweet': tweet,
            'prediction': 'Disaster' if pred == 1 else 'Non-Disaster',
            'probability': float(prob[1]),
            'confidence': float(prob[1]) if pred == 1 else float(prob[0])
        }
        results.append(result)
    
    return jsonify({'results': results})

@app.route('/stats')
def stats():
    """Return model statistics."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Try to load comparison results
    try:
        comparison_df = pd.read_csv('model/comparison_results.csv')
        stats = comparison_df.to_dict('records')
    except:
        stats = []
    
    return jsonify({
        'model_loaded': True,
        'model_type': 'Logistic Regression with TF-IDF',
        'comparison_stats': stats
    })

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    import os
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tweet Disaster Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        .form-group textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e1e1;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        
        .form-group textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            font-weight: 600;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
            border-left: 5px solid #667eea;
            display: none;
        }
        
        .result.show {
            display: block;
            animation: slideIn 0.3s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .result-item {
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        
        .result-item strong {
            color: #667eea;
        }
        
        .prediction-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .prediction-badge.disaster {
            background: #ff6b6b;
            color: white;
        }
        
        .prediction-badge.non-disaster {
            background: #51cf66;
            color: white;
        }
        
        .confidence-bar {
            height: 10px;
            background: #e1e1e1;
            border-radius: 5px;
            margin-top: 5px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease-out;
        }
        
        .examples {
            margin-top: 40px;
        }
        
        .examples h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .example-tweets {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }
        
        .example-tweet {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .example-tweet:hover {
            background: #e9ecef;
        }
        
        .stats {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .stats h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚨 Tweet Disaster Classifier</h1>
            <p>Classify tweets as disaster-related or not</p>
        </div>
        
        <div class="content">
            <form id="predictionForm">
                <div class="form-group">
                    <label for="tweet">Enter a Tweet:</label>
                    <textarea id="tweet" name="tweet" placeholder="Enter a tweet to classify..." required></textarea>
                </div>
                
                <div class="form-group">
                    <button type="submit" class="btn">Classify Tweet</button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Classifying...</p>
            </div>
            
            <div class="result" id="result"></div>
            
            <div class="examples">
                <h3>📝 Try These Examples:</h3>
                <div class="example-tweets">
                    <div class="example-tweet" onclick="useExample('Earthquake hits California, massive damage reported')">Earthquake hits California, massive damage reported</div>
                    <div class="example-tweet" onclick="useExample('Just had a great lunch with friends')">Just had a great lunch with friends</div>
                    <div class="example-tweet" onclick="useExample('Forest fire spreading rapidly, evacuate immediately')">Forest fire spreading rapidly, evacuate immediately</div>
                    <div class="example-tweet" onclick="useExample('The weather is nice today')">The weather is nice today</div>
                    <div class="example-tweet" onclick="useExample('Tornado warning issued for Oklahoma')">Tornado warning issued for Oklahoma</div>
                    <div class="example-tweet" onclick="useExample('Happy birthday to my best friend!')">Happy birthday to my best friend!</div>
                </div>
            </div>
            
            <div class="stats">
                <h3>📊 Model Information</h3>
                <p><strong>Model Type:</strong> Logistic Regression with TF-IDF</p>
                <p><strong>Features:</strong> 5000 TF-IDF features with n-grams</p>
                <p><strong>Training Samples:</strong> ~6800 tweets</p>
                <p><strong>Expected Accuracy:</strong> ~85-90%</p>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const tweet = document.getElementById('tweet').value;
            if (!tweet.trim()) {
                alert('Please enter a tweet');
                return;
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('result').classList.remove('show');
            
            // Send request to server
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ tweet: tweet })
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').classList.remove('show');
                
                // Display result
                const resultDiv = document.getElementById('result');
                const predictionClass = data.prediction === 'Disaster' ? 'disaster' : 'non-disaster';
                const confidencePercent = (data.confidence * 100).toFixed(2);
                
                resultDiv.innerHTML = `
                    <h3>🎯 Classification Result</h3>
                    <div class="result-item">
                        <strong>Tweet:</strong> ${data.tweet}
                    </div>
                    <div class="result-item">
                        <strong>Prediction:</strong> 
                        <span class="prediction-badge ${predictionClass}">${data.prediction}</span>
                    </div>
                    <div class="result-item">
                        <strong>Confidence:</strong> ${confidencePercent}%
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                        </div>
                    </div>
                    <div class="result-item">
                        <strong>Disaster Probability:</strong> ${(data.probability * 100).toFixed(2)}%
                    </div>
                `;
                
                resultDiv.classList.add('show');
            })
            .catch(error => {
                // Hide loading
                document.getElementById('loading').classList.remove('show');
                
                console.error('Error:', error);
                alert('An error occurred while classifying the tweet');
            });
        });
        
        function useExample(tweet) {
            document.getElementById('tweet').value = tweet;
            document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
        }
    </script>
</body>
</html>''')
    
    app.run(debug=True, host='0.0.0.0', port=5000)