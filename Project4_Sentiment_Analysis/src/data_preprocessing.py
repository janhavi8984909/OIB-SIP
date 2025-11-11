import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class TextPreprocessor:
    """
    A comprehensive text preprocessing class for Twitter sentiment analysis
    """
    
    def __init__(self, download_nltk=True):
        """
        Initialize the text preprocessor
        
        Args:
            download_nltk (bool): Whether to download NLTK data
        """
        if download_nltk:
            self.download_nltk_resources()
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = None
    
    def download_nltk_resources(self):
        """Download required NLTK resources"""
        resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.download(resource, quiet=True)
            except:
                print(f"Warning: Could not download {resource}")
    
    def clean_text(self, text):
        """
        Clean and preprocess text data
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, text_column='clean_text', target_column='category'):
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            target_column (str): Name of target column
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_clean = df.copy()
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        print(f"Removed {initial_count - len(df_clean)} duplicate rows")
        
        # Handle missing values
        df_clean = df_clean.dropna(subset=[text_column, target_column])
        print(f"After handling missing values: {len(df_clean)} rows")
        
        # Clean text
        print("Cleaning text data...")
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
        print(f"Final dataset size: {len(df_clean)} rows")
        
        return df_clean
    def create_features(self, texts, method='tfidf', **kwargs):
        """
        Create feature vectors from text
        
        Args:
            texts (list): List of text documents
            method (str): Feature extraction method ('tfidf' or 'count')
            **kwargs: Additional parameters for vectorizer
            
        Returns:
            scipy.sparse matrix: Feature matrix
        """
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(**kwargs)
        elif method == 'count':
            self.vectorizer = CountVectorizer(**kwargs)
        else:
            raise ValueError("Method must be 'tfidf' or 'count'")
        
        features = self.vectorizer.fit_transform(texts)
        return features
    
    def prepare_train_test_split(self, df, text_column='cleaned_text', target_column='category', 
                               test_size=0.2, random_state=42):
        """
        Prepare train-test split
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of text column
            target_column (str): Name of target column
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df[text_column]
        y = df[target_column]
        
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )