import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
import string

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TwitterSentimentAnalysis:
    def __init__(self, data_path):
        """
        Initialize the sentiment analysis class
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.best_model = None
        
    def load_data(self):
        """
        Load and explore the dataset
        """
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        
        # Display basic information
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst few rows:")
        print(self.df.head())
        
        print("\nDataset info:")
        print(self.df.info())
        
        print("\nSentiment distribution:")
        print(self.df['category'].value_counts())
        
        return self.df
    
    def explore_data(self):
        """
        Perform exploratory data analysis
        """
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Sentiment distribution
        plt.figure(figsize=(10, 6))
        sentiment_counts = self.df['category'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        plt.subplot(1, 2, 1)
        plt.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.pie(sentiment_counts.values, labels=['Negative', 'Neutral', 'Positive'], 
                autopct='%1.1f%%', colors=colors)
        plt.title('Sentiment Proportion')
        
        plt.tight_layout()
        plt.show()
        
        # Text length analysis
        self.df['text_length'] = self.df['clean_text'].apply(len)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x='category', y='text_length', data=self.df)
        plt.title('Text Length by Sentiment')
        
        plt.subplot(1, 2, 2)
        sns.histplot(data=self.df, x='text_length', hue='category', multiple="stack")
        plt.title('Text Length Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Word clouds for each sentiment
        self.generate_wordclouds()
    
    def generate_wordclouds(self):
        """
        Generate word clouds for each sentiment category
        """
        sentiments = [-1, 0, 1]
        sentiment_names = { -1: 'Negative', 0: 'Neutral', 1: 'Positive' }
        
        plt.figure(figsize=(15, 5))
        
        for i, sentiment in enumerate(sentiments):
            plt.subplot(1, 3, i+1)
            text = ' '.join(self.df[self.df['category'] == sentiment]['clean_text'])
            wordcloud = WordCloud(width=400, height=300, 
                                background_color='white',
                                max_words=100,
                                colormap='viridis').generate(text)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'{sentiment_names[sentiment]} Sentiment')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def preprocess_text(self, text):
        """
        Preprocess text data: cleaning, tokenization, lemmatization
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def prepare_data(self):
        """
        Prepare data for model training
        """
        print("\n=== DATA PREPARATION ===")
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Preprocess text
        print("Preprocessing text data...")
        self.df['cleaned_text'] = self.df['clean_text'].apply(self.preprocess_text)
        
        # Split data
        X = self.df['cleaned_text']
        y = self.df['category']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(self.X_train)}")
        print(f"Test set size: {len(self.X_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        print("\n=== MODEL TRAINING ===")
        
        # Define models and parameters for grid search
        models = {
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {
                    'vectorizer__max_features': [1000, 2000, 3000],
                    'classifier__alpha': [0.1, 0.5, 1.0]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'params': {
                    'vectorizer__max_features': [1000, 2000, 3000],
                    'classifier__C': [0.1, 1, 10],
                    'classifier__max_iter': [1000]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'vectorizer__max_features': [1000, 2000],
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20]
                }
            }
        }
        
        best_score = 0
        self.best_model = None
        
        for name, model_info in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline
            pipeline = Pipeline([
                ('vectorizer', TfidfVectorizer()),
                ('classifier', model_info['model'])
            ])
            
            # Grid search
            grid_search = GridSearchCV(
                pipeline, 
                model_info['params'], 
                cv=5, 
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Store model
            self.models[name] = {
                'pipeline': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
            
            # Update best model
            if grid_search.best_score_ > best_score:
                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
        
        return self.models, self.best_model
    
    def evaluate_models(self):
        """
        Evaluate all trained models
        """
        print("\n=== MODEL EVALUATION ===")
        
        results = {}
        
        for name, model_info in self.models.items():
            print(f"\n{name} Evaluation:")
            
            # Predictions
            y_pred = model_info['pipeline'].predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
            
            # Confusion Matrix
            plt.figure(figsize=(6, 4))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Neutral', 'Positive'],
                       yticklabels=['Negative', 'Neutral', 'Positive'])
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.show()
        
        return results
    
    def compare_models(self, results):
        """
        Compare performance of all models
        """
        print("\n=== MODEL COMPARISON ===")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        print(comparison_df)
        
        # Visualization
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_df['Model'], comparison_df['Accuracy'], color='skyblue')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add accuracy values on bars
        for i, v in enumerate(comparison_df['Accuracy']):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for new text using the best model
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Predict sentiment
        prediction = self.best_model.predict([cleaned_text])[0]
        probability = self.best_model.predict_proba([cleaned_text])[0]
        
        sentiment_map = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        
        result = {
            'text': text,
            'cleaned_text': cleaned_text,
            'sentiment': sentiment_map[prediction],
            'confidence': max(probability),
            'probabilities': {
                'Negative': probability[0],
                'Neutral': probability[1],
                'Positive': probability[2]
            }
        }
        
        return result
    
    def run_complete_analysis(self):
        """
        Run the complete sentiment analysis pipeline
        """
        print("Starting Twitter Sentiment Analysis...")
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Explore data
        self.explore_data()
        
        # Step 3: Prepare data
        self.prepare_data()
        
        # Step 4: Train models
        self.train_models()
        
        # Step 5: Evaluate models
        results = self.evaluate_models()
        
        # Step 6: Compare models
        comparison_df = self.compare_models(results)
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Best model: {type(self.best_model.named_steps['classifier']).__name__}")
        print(f"Best accuracy: {comparison_df['Accuracy'].iloc[0]:.4f}")
        
        return self.best_model, results

def main():
    """
    Main function to run the sentiment analysis
    """
    # Initialize the analyzer
    analyzer = TwitterSentimentAnalysis('Twitter_Data.csv')
    
    # Run complete analysis
    best_model, results = analyzer.run_complete_analysis()
    
    # Example predictions
    print("\n=== EXAMPLE PREDICTIONS ===")
    test_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is okay, nothing special but gets the job done.",
        "Terrible experience. The service was horrible and I'm very disappointed."
    ]
    
    for text in test_texts:
        result = analyzer.predict_sentiment(text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Probabilities: {result['probabilities']}")

if __name__ == "__main__":
    main()