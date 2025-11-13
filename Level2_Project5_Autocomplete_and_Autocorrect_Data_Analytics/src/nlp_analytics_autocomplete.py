import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import logging
from .text_preprocessing import TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoComplete:
    """Autocomplete system using n-gram language models"""
    
    def __init__(self, max_suggestions: int = 5, context_window: int = 3):
        self.max_suggestions = max_suggestions
        self.context_window = context_window
        self.ngram_models = {}
        self.vocabulary = set()
        self.preprocessor = TextPreprocessor()
        self.is_trained = False
        
        logger.info("Autocomplete system initialized")
    
    def train(self, texts: List[str], n_range: Tuple[int, int] = (1, 3)):
        """Train the autocomplete model on text corpus"""
        logger.info(f"Training autocomplete model on {len(texts)} texts...")
        
        # Preprocess texts
        processed_texts = self.preprocessor.preprocess_corpus(texts)
        
        # Build vocabulary
        vocab_dict = self.preprocessor.build_vocabulary(processed_texts)
        self.vocabulary = set(vocab_dict.keys())
        
        # Create n-gram models
        self.ngram_models = self.preprocessor.create_ngram_model(
            processed_texts, n_range
        )
        
        self.is_trained = True
        logger.info("Autocomplete model training completed")
    
    def get_suggestions(self, input_text: str) -> List[Tuple[str, float]]:
        """Get word suggestions based on input text"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting suggestions")
        
        # Preprocess input
        processed_input = self.preprocessor.preprocess_text(input_text)
        words = processed_input.split()
        
        # Use last few words as context
        context = words[-self.context_window:]
        
        suggestions = []
        
        # Try different n-gram levels
        for n in sorted(self.ngram_models.keys(), reverse=True):
            if len(context) >= n - 1:
                context_ngram = ' '.join(context[-(n-1):]) if n > 1 else ""
                suggestions.extend(self._get_ngram_suggestions(context_ngram, n))
        
        # Remove duplicates and sort by probability
        unique_suggestions = {}
        for word, prob in suggestions:
            if word not in unique_suggestions or prob > unique_suggestions[word]:
                unique_suggestions[word] = prob
        
        # Sort by probability and return top suggestions
        sorted_suggestions = sorted(
            unique_suggestions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_suggestions]
        
        return sorted_suggestions
    
    def _get_ngram_suggestions(self, context: str, n: int) -> List[Tuple[str, float]]:
        """Get suggestions for specific n-gram level"""
        suggestions = []
        ngram_model = self.ngram_models[n]
        
        # Calculate total count for context
        context_total = 0
        for ngram, count in ngram_model.items():
            if ngram.startswith(context + ' ') if context else True:
                context_total += count
        
        if context_total == 0:
            return suggestions
        
        # Calculate probabilities for each possible next word
        for ngram, count in ngram_model.items():
            ngram_words = ngram.split()
            
            if n == 1:  # Unigram
                next_word = ngram_words[0]
                prob = count / context_total
                suggestions.append((next_word, prob))
            
            else:  # Bigram or higher
                if context == ' '.join(ngram_words[:-1]):
                    next_word = ngram_words[-1]
                    prob = count / context_total
                    suggestions.append((next_word, prob))
        
        return suggestions
    
    def predict_next_word(self, input_text: str) -> str:
        """Predict the most likely next word"""
        suggestions = self.get_suggestions(input_text)
        return suggestions[0][0] if suggestions else ""
    
    def evaluate_suggestions(self, test_cases: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate autocomplete performance on test cases"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        correct_predictions = 0
        total_predictions = 0
        
        for context, expected_next in test_cases:
            suggestions = self.get_suggestions(context)
            suggested_words = [word for word, prob in suggestions]
            
            if expected_next in suggested_words:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }

class TrieNode:
    """Trie node for efficient prefix-based searching"""
    
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0

class TrieAutoComplete:
    """Trie-based autocomplete for efficient prefix searching"""
    
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word: str, frequency: int = 1):
        """Insert a word into the trie"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.frequency = frequency
    
    def search(self, prefix: str) -> List[Tuple[str, int]]:
        """Search for words with given prefix"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        return self._get_all_words_from_node(node, prefix)
    
    def _get_all_words_from_node(self, node: TrieNode, prefix: str) -> List[Tuple[str, int]]:
        """Get all words from a trie node"""
        words = []
        if node.is_end_of_word:
            words.append((prefix, node.frequency))
        
        for char, child_node in node.children.items():
            words.extend(self._get_all_words_from_node(child_node, prefix + char))
        
        return sorted(words, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    # Test the autocomplete system
    ac = AutoComplete()
    
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables computers to understand human language",
        "data science involves statistics programming and domain knowledge",
        "deep learning models require large amounts of training data"
    ]
    
    ac.train(sample_texts)
    
    test_inputs = [
        "the quick",
        "machine",
        "natural language",
        "deep"
    ]
    
    for test_input in test_inputs:
        suggestions = ac.get_suggestions(test_input)
        print(f"Input: '{test_input}'")
        print(f"Suggestions: {suggestions}")
        print()
        
                                        