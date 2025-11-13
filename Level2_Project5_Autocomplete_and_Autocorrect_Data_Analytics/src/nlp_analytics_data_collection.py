import requests
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCollector:
    """Data collector for NLP text datasets"""
    
    def __init__(self, data_path: str = "01_data/raw/nlp_datasets"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
    
    def download_sample_datasets(self):
        """Download sample text datasets for demonstration"""
        logger.info("Downloading sample NLP datasets...")
        
        # Sample datasets URLs (replace with actual URLs in production)
        datasets = {
            'wikipedia_sample': 'https://raw.githubusercontent.com/datasets/sample-data/master/sample.txt',
            'news_headlines': 'https://raw.githubusercontent.com/datasets/sample-data/master/news.txt'
        }
        
        for name, url in datasets.items():
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    file_path = self.data_path / f"{name}.txt"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    logger.info(f"Downloaded {name} to {file_path}")
                else:
                    logger.warning(f"Failed to download {name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading {name}: {str(e)}")
    
    def load_text_files(self, directory: str = None) -> List[str]:
        """Load all text files from directory"""
        if directory is None:
            directory = self.data_path
        
        text_files = list(Path(directory).glob("*.txt"))
        all_texts = []
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_texts.append(text)
                logger.info(f"Loaded {file_path} - {len(text)} characters")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        return all_texts
    
    def create_sample_corpus(self) -> List[str]:
        """Create a sample corpus if no external data is available"""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Natural language processing enables computers to understand human language",
            "Data science involves statistics, programming, and domain knowledge",
            "Deep learning models require large amounts of training data",
            "Python is a popular programming language for data analysis",
            "Credit card fraud detection uses machine learning algorithms",
            "Autocomplete systems predict words based on context",
            "Autocorrect functionality fixes spelling errors in real time",
            "User experience is crucial for text prediction systems"
        ]
        
        # Add some variations and errors for testing
        sample_texts_with_errors = sample_texts + [
            "The quik brown fox jumps ovr the lazy dog",
            "Machine lurning is a subset of artifical intelligence",
            "Naturl language processing enbles computers to understand human language"
        ]
        
        return sample_texts_with_errors
    
    def save_corpus(self, texts: List[str], filename: str = "corpus.txt"):
        """Save text corpus to file"""
        file_path = self.data_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        logger.info(f"Saved corpus to {file_path} - {len(texts)} texts")

if __name__ == "__main__":
    collector = DataCollector()
    
    # Try to download datasets
    collector.download_sample_datasets()
    
    # Load or create corpus
    texts = collector.load_text_files()
    if not texts:
        texts = collector.create_sample_corpus()
        collector.save_corpus(texts)
    
    print(f"Loaded {len(texts)} text documents")
    