# 🚨 Tweet Disaster Classification Model

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)

A machine learning model to classify tweets as disaster-related or not. This project uses NLP techniques with TF-IDF vectorization and various classification algorithms to accurately identify disaster-related content on Twitter.

## 🎯 Project Overview

This project implements a comprehensive tweet disaster classification system that can:
- Preprocess and clean tweet text
- Extract meaningful features using TF-IDF
- Train multiple classification models
- Compare model performance
- Make predictions on new tweets
- Serve predictions via a web interface

## 📁 Project Structure

```
Project-1/
├── data/
│   └── twitter_disaster.csv          # Dataset
├── model/
│   ├── tweet_disaster_classifier.py  # Main model training script
│   ├── best_logistic_regression.pkl  # Trained model (generated)
│   ├── best_random_forest.pkl        # Alternative model (generated)
│   ├── tweet_disaster_predictor.pkl   # Prediction pipeline (generated)
│   ├── comparison_results.csv        # Model comparison (generated)
│   └── plots/                         # Visualizations (generated)
├── notebooks/
│   └── Tweet_Disaster_Classification.ipynb  # Jupyter notebook
├── app.py                            # Flask web application
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone the repository (if not already cloned)
git clone https://github.com/Bijender2233/Project-1.git
cd Project-1

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Run the main training script
python model/tweet_disaster_classifier.py
```

This will:
- Load and explore the dataset
- Preprocess the text data
- Extract TF-IDF features
- Train and compare multiple models
- Perform hyperparameter tuning
- Save the best model
- Create visualizations

### 3. Run the Web Application

```bash
# Start the Flask web application
python app.py
```

Then open your browser to: [http://localhost:5000](http://localhost:5000)

### 4. Use the Jupyter Notebook

```bash
# Start Jupyter notebook
jupyter notebook
```

Then open: [http://localhost:8888](http://localhost:8888) and navigate to `notebooks/Tweet_Disaster_Classification.ipynb`

## 📊 Dataset

The dataset contains tweets labeled as disaster-related (1) or not (0):

- **Total samples**: ~7,600 tweets
- **Features**: id, keyword, location, text, target
- **Target**: Binary (1 = disaster, 0 = non-disaster)
- **Class distribution**: Approximately balanced

### Sample Data

| id | keyword | location | text | target |
|----|---------|----------|------|--------|
| 1 | | | Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all | 1 |
| 4 | | | Forest fire near La Ronge Sask. Canada | 1 |
| 23 | | | What's up man? | 0 |
| 24 | | | I love fruits | 0 |

## 🔧 Model Training

### Text Preprocessing

The model performs comprehensive text cleaning:
- Convert to lowercase
- Remove URLs, user mentions, and hashtags
- Remove punctuation and numbers
- Tokenize text
- Remove stopwords
- Lemmatize words

### Feature Extraction

- **TF-IDF Vectorization**: Converts text to numerical features
- **N-grams**: Includes unigrams and bigrams
- **Feature selection**: Maximum 5,000 features
- **Document frequency filtering**: Ignore rare and too common terms

### Classification Models

The following models are trained and compared:

1. **Logistic Regression** - Best performing model
2. **Naive Bayes** - Fast and simple baseline
3. **Support Vector Machine (SVM)** - Powerful linear classifier
4. **Random Forest** - Ensemble method
5. **Gradient Boosting** - Advanced ensemble method
6. **K-Nearest Neighbors (KNN)** - Instance-based learning

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.88 | ~0.87 | ~0.88 | ~0.88 |
| Random Forest | ~0.85 | ~0.84 | ~0.85 | ~0.84 |
| SVM | ~0.87 | ~0.86 | ~0.87 | ~0.86 |
| Naive Bayes | ~0.82 | ~0.81 | ~0.82 | ~0.81 |

## 🎯 Usage Examples

### Python API

```python
from model.tweet_disaster_classifier import TweetDisasterClassifier

# Initialize and train
classifier = TweetDisasterClassifier(data_path='data/twitter_disaster.csv')
classifier.load_data()
classifier.prepare_data()
classifier.extract_features()
classifier.train_model('logistic_regression')

# Make predictions
result = classifier.predict("Earthquake hits California, massive damage reported")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

### Command Line

```bash
# Train the model
python model/tweet_disaster_classifier.py

# Run web application
python app.py
```

### Web Interface

1. Start the Flask app: `python app.py`
2. Open browser to: `http://localhost:5000`
3. Enter a tweet and click "Classify Tweet"
4. View the prediction result

## 🌐 Web Application Features

- **Interactive Interface**: Clean, modern UI for tweet classification
- **Real-time Prediction**: Instant classification results
- **Confidence Visualization**: Visual confidence bar
- **Example Tweets**: Pre-loaded examples to try
- **Responsive Design**: Works on mobile and desktop

## 📈 Model Evaluation

The model is evaluated using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **Confusion Matrix**: Visual representation of predictions

## 🔬 Hyperparameter Tuning

The model performs grid search for optimal parameters:

```python
# Example parameter grid for Logistic Regression
param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
```

## 📊 Visualizations

The project generates various visualizations:

- **Target Distribution**: Class balance visualization
- **Text Length Distribution**: Analysis of tweet lengths
- **Confusion Matrix**: Model performance visualization
- **ROC Curve**: Receiver Operating Characteristic
- **Word Clouds**: Most common words in each class
- **Model Comparison**: Performance comparison across models

## 🛠️ Customization

### Change Dataset

Replace the dataset file at `data/twitter_disaster.csv` with your own dataset. The CSV should have at least two columns: `text` (tweet content) and `target` (0 or 1).

### Modify Preprocessing

Edit the `preprocess_text()` method in `tweet_disaster_classifier.py` to customize text cleaning.

### Add New Models

Add new classifiers to the `train_model()` method:

```python
# Add to the models dictionary
models = {
    # ... existing models
    'new_model': NewClassifier,
}
```

### Adjust Hyperparameters

Modify the parameter grids in the `hyperparameter_tuning()` method.

## 📦 Dependencies

### Required
- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn
- joblib
- flask

### Optional
- jupyter (for notebooks)
- wordcloud (for word cloud visualizations)
- tensorflow (for deep learning extensions)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: Twitter Disaster Tweets (Kaggle)
- Libraries: scikit-learn, NLTK, pandas, matplotlib, seaborn, Flask
- Inspiration: Natural Language Processing and Machine Learning communities

## 📞 Contact

For questions or support, please contact:
- **Author**: Bijender
- **GitHub**: [Bijender2233](https://github.com/Bijender2233)
- **Project**: [Project-1](https://github.com/Bijender2233/Project-1)

---

**Made with ❤️ and Python**

*Last updated: 2024*