import re
from collections import Counter
from typing import List, Dict, Any, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoCorrect:
    """Autocorrect system using statistical methods"""
    
    def __init__(self, max_edit_distance: int = 2, prefix_length: int = 7):
        self.max_edit_distance = max_edit_distance
        self.prefix_length = prefix_length
        self.word_frequency = Counter()
        self.vocabulary = set()
        self.is_trained = False
        
        logger.info("Autocorrect system initialized")
    
    def train(self, texts: List[str]):
        """Train the autocorrect model on text corpus"""
        logger.info(f"Training autocorrect model on {len(texts)} texts...")
        
        # Extract words from texts
        all_words = []
        for text in texts:
            words = self._tokenize_text(text)
            all_words.extend(words)
        
        # Build vocabulary and frequency dictionary
        self.word_frequency = Counter(all_words)
        self.vocabulary = set(self.word_frequency.keys())
        
        self.is_trained = True
        logger.info(f"Autocorrect model trained with {len(self.vocabulary)} words")
    
    def correct(self, word: str) -> str:
        """Correct a single word"""
        if not self.is_trained:
            raise ValueError("Model must be trained before correction")
        
        # If word is in vocabulary, return as is
        if word in self.vocabulary:
            return word
        
        # Find candidate corrections
        candidates = self._get_candidates(word)
        
        if not candidates:
            return word  # Return original if no candidates found
        
        # Return the candidate with highest frequency
        return max(candidates, key=lambda x: self.word_frequency[x])
    
    def _get_candidates(self, word: str) -> Set[str]:
        """Get candidate corrections for a word"""
        candidates = set()
        
        # Generate edits and filter valid words
        edits = self._generate_edits(word, self.max_edit_distance)
        valid_edits = [e for e in edits if e in self.vocabulary]
        
        if valid_edits:
            candidates.update(valid_edits)
        
        # If no valid edits found, try more distant edits
        if not candidates:
            distant_edits = self._generate_edits(word, self.max_edit_distance + 1)
            valid_distant_edits = [e for e in distant_edits if e in self.vocabulary]
            candidates.update(valid_distant_edits)
        
        return candidates
    
    def _generate_edits(self, word: str, max_distance: int) -> Set[str]:
        """Generate all possible edits within maximum edit distance"""
        edits = set([word])
        
        for _ in range(max_distance):
            new_edits = set()
            for edit in edits:
                new_edits.update(self._generate_single_edits(edit))
            edits.update(new_edits)
        
        return edits
    
    def _generate_single_edits(self, word: str) -> Set[str]:
        """Generate all possible single-edits for a word"""
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        
        return set(deletes + transposes + replaces + inserts)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and split by non-alphanumeric characters
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def evaluate_correction(self, test_cases: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate autocorrect performance on test cases"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        correct_corrections = 0
        total_cases = len(test_cases)
        
        for incorrect, expected in test_cases:
            correction = self.correct(incorrect)
            if correction == expected:
                correct_corrections += 1
        
        accuracy = correct_corrections / total_cases if total_cases > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct_corrections': correct_corrections,
            'total_cases': total_cases
        }

class SymSpellAutoCorrect:
    """SymSpell algorithm for fast spelling correction"""
    
    def __init__(self, max_edit_distance: int = 2):
        self.max_edit_distance = max_edit_distance
        self.dictionary = {}
        self.is_trained = False
    
    def train(self, texts: List[str]):
        """Train SymSpell model"""
        logger.info("Training SymSpell model...")
        
        # Build frequency dictionary
        word_frequency = Counter()
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            word_frequency.update(words)
        
        # Create dictionary with deletes
        for word, freq in word_frequency.items():
            self.dictionary[word] = freq
            
            # Generate deletes
            edits = self._generate_deletes(word, self.max_edit_distance)
            for edit in edits:
                if edit in self.dictionary:
                    # Keep the word with highest frequency
                    if freq > self.dictionary[edit]:
                        self.dictionary[edit] = freq
                else:
                    self.dictionary[edit] = freq
        
        self.is_trained = True
        logger.info(f"SymSpell model trained with {len(self.dictionary)} entries")
    
    def _generate_deletes(self, word: str, max_distance: int) -> Set[str]:
        """Generate delete edits for a word"""
        deletes = set()
        
        for distance in range(1, max_distance + 1):
            if distance == 1:
                # Single deletes
                for i in range(len(word)):
                    deletes.add(word[:i] + word[i+1:])
            else:
                # Recursive deletes for higher distances
                current_deletes = set(deletes)
                for delete in current_deletes:
                    for i in range(len(delete)):
                        new_delete = delete[:i] + delete[i+1:]
                        deletes.add(new_delete)
        
        return deletes
    
    def correct(self, word: str) -> str:
        """Correct a word using SymSpell"""
        if not self.is_trained:
            raise ValueError("Model must be trained before correction")
        
        if word in self.dictionary:
            return word
        
        # Look for suggestions
        suggestions = {}
        
        # Generate edits and look in dictionary
        edits = self._generate_deletes(word, self.max_edit_distance)
        for edit in edits:
            if edit in self.dictionary:
                suggestions[edit] = self.dictionary[edit]
        
        if suggestions:
            return max(suggestions.items(), key=lambda x: x[1])[0]
        
        return word  # Return original if no suggestions found

if __name__ == "__main__":
    # Test the autocorrect system
    corrector = AutoCorrect()
    
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables computers to understand human language",
        "data science involves statistics programming and domain knowledge"
    ]
    
    corrector.train(sample_texts)
    
    test_words = [
        ("quik", "quick"),
        ("browm", "brown"),
        ("jumpes", "jumps"),
        ("mashine", "machine"),
        ("lurn", "learning"),
        ("procesing", "processing"),
        ("computrs", "computers")
    ]
    
    print("Autocorrect Test Results:")
    for incorrect, expected in test_words:
        correction = corrector.correct(incorrect)
        status = "✓" if correction == expected else "✗"
        print(f"{status} '{incorrect}' -> '{correction}' (expected: '{expected}')")
    
    # Test SymSpell
    print("\nSymSpell Test:")
    sym_corrector = SymSpellAutoCorrect()
    sym_corrector.train(sample_texts)
    
    for incorrect, expected in test_words:
        correction = sym_corrector.correct(incorrect)
        status = "✓" if correction == expected else "✗"
        print(f"{status} '{incorrect}' -> '{correction}' (expected: '{expected}')")
        