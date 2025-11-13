import re
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from typing import List, Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Text preprocessing for NLP tasks"""
    
    def __init__(self, 
                 lower_case: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = True,
                 stem_words: bool = False,
                 lemmatize_words: bool = True):
        
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stem_words = stem_words
        self.lemmatize_words = lemmatize_words
        
        # Initialize NLP tools
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if stem_words else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize_words else None
        
        logger.info("Text preprocessor initialized")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess a single text string"""
        # Convert to lowercase
        if self.lower_case:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming
        if self.stem_words and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Lemmatization
        if self.lemmatize_words and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts"""
        logger.info(f"Preprocessing {len(texts)} texts...")
        
        processed_texts = []
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(texts)} texts")
            
            processed_text = self.preprocess_text(text)
            processed_texts.append(processed_text)
        
        logger.info("Text preprocessing completed")
        return processed_texts
    
    def build_vocabulary(self, texts: List[str], min_frequency: int = 2) -> Dict[str, int]:
        """Build vocabulary from processed texts"""
        logger.info("Building vocabulary...")
        
        # Combine all texts and split into words
        all_words = []
        for text in texts:
            words = text.split()
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter by minimum frequency
        vocabulary = {word: count for word, count in word_counts.items() 
                     if count >= min_frequency}
        
        # Sort by frequency
        vocabulary = dict(sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
        
        logger.info(f"Vocabulary built: {len(vocabulary)} unique words "
                   f"(min frequency: {min_frequency})")
        
        return vocabulary
    
    def generate_ngrams(self, texts: List[str], n: int = 2) -> List[List[str]]:
        """Generate n-grams from texts"""
        ngrams_list = []
        
        for text in texts:
            words = text.split()
            if len(words) >= n:
                for i in range(len(words) - n + 1):
                    ngram = words[i:i + n]
                    ngrams_list.append(ngram)
        
        return ngrams_list
    
    def create_ngram_model(self, texts: List[str], n_range: Tuple[int, int] = (1, 3)) -> Dict[int, Counter]:
        """Create n-gram language model"""
        logger.info(f"Creating n-gram model for n={n_range[0]} to {n_range[1]}")
        
        ngram_models = {}
        
        for n in range(n_range[0], n_range[1] + 1):
            ngrams = self.generate_ngrams(texts, n)
            ngram_counts = Counter([' '.join(ngram) for ngram in ngrams])
            ngram_models[n] = ngram_counts
            
            logger.info(f"Generated {len(ngram_counts)} {n}-grams")
        
        return ngram_models

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog!",
        "Machine learning is amazing for natural language processing.",
        "This is a test sentence with some numbers 123 and punctuation!!!"
    ]
    
    processed = preprocessor.preprocess_corpus(sample_texts)
    vocab = preprocessor.build_vocabulary(processed)
    
    print("Processed texts:")
    for i, text in enumerate(processed):
        print(f"{i+1}: {text}")
    
    print(f"\nVocabulary ({len(vocab)} words):")
    for word, count in list(vocab.items())[:10]:
        print(f"  {word}: {count}")
        