import time
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from .autocomplete import AutoComplete
from .autocorrect import AutoCorrect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NLPEvaluator:
    """Comprehensive evaluation for NLP autocomplete and autocorrect systems"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_autocomplete(self, autocomplete: AutoComplete, 
                            test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Evaluate autocomplete system performance"""
        logger.info("Evaluating autocomplete system...")
        
        start_time = time.time()
        
        # Accuracy metrics
        accuracy_results = autocomplete.evaluate_suggestions(test_cases)
        
        # Response time metrics
        response_times = []
        for context, expected in test_cases:
            start_pred = time.time()
            autocomplete.get_suggestions(context)
            end_pred = time.time()
            response_times.append((end_pred - start_pred) * 1000)  # Convert to ms
        
        # Keystroke savings calculation
        keystroke_savings = self._calculate_keystroke_savings(autocomplete, test_cases)
        
        evaluation_time = time.time() - start_time
        
        results = {
            'accuracy': accuracy_results['accuracy'],
            'correct_predictions': accuracy_results['correct_predictions'],
            'total_predictions': accuracy_results['total_predictions'],
            'response_time_ms': {
                'mean': np.mean(response_times),
                'std': np.std(response_times),
                'min': np.min(response_times),
                'max': np.max(response_times)
            },
            'keystroke_savings': keystroke_savings,
            'evaluation_time_seconds': evaluation_time
        }
        
        self.results['autocomplete'] = results
        return results
    
    def evaluate_autocorrect(self, autocorrect: AutoCorrect, 
                           test_cases: List[Tuple[str, str]]) -> Dict[str, Any]:
        """Evaluate autocorrect system performance"""
        logger.info("Evaluating autocorrect system...")
        
        start_time = time.time()
        
        # Accuracy metrics
        accuracy_results = autocorrect.evaluate_correction(test_cases)
        
        # Response time metrics
        response_times = []
        for incorrect, expected in test_cases:
            start_correct = time.time()
            autocorrect.correct(incorrect)
            end_correct = time.time()
            response_times.append((end_correct - start_correct) * 1000)  # Convert to ms
        
        # Precision and recall for error detection
        precision_recall = self._calculate_precision_recall(autocorrect, test_cases)
        
        evaluation_time = time.time() - start_time
        
        results = {
            'accuracy': accuracy_results['accuracy'],
            'correct_corrections': accuracy_results['correct_corrections'],
            'total_cases': accuracy_results['total_cases'],
            'response_time_ms': {
                'mean': np.mean(response_times),
                'std': np.std(response_times),
                'min': np.min(response_times),
                'max': np.max(response_times)
            },
            'precision': precision_recall['precision'],
            'recall': precision_recall['recall'],
            'f1_score': precision_recall['f1_score'],
            'evaluation_time_seconds': evaluation_time
        }
        
        self.results['autocorrect'] = results
        return results
    
    def _calculate_keystroke_savings(self, autocomplete: AutoComplete, 
                                   test_cases: List[Tuple[str, str]]) -> Dict[str, float]:
        """Calculate keystroke savings for autocomplete"""
        total_keystrokes_without = 0
        total_keystrokes_with = 0
        
        for context, expected_next in test_cases:
            # Keystrokes without autocomplete (typing full word)
            keystrokes_without = len(expected_next)
            
            # Keystrokes with autocomplete
            suggestions = autocomplete.get_suggestions(context)
            if suggestions:
                # Assume user accepts first suggestion after typing 1 character
                keystrokes_with = 1
            else:
                keystrokes_with = keystrokes_without
            
            total_keystrokes_without += keystrokes_without
            total_keystrokes_with += keystrokes_with
        
        savings_ratio = (total_keystrokes_without - total_keystrokes_with) / total_keystrokes_without
        savings_percentage = savings_ratio * 100
        
        return {
            'savings_ratio': savings_ratio,
            'savings_percentage': savings_percentage,
            'total_keystrokes_without': total_keystrokes_without,
            'total_keystrokes_with': total_keystrokes_with
        }
    
    def _calculate_precision_recall(self, autocorrect: AutoCorrect, 
                                  test_cases: List[Tuple[str, str]]) -> Dict[str, float]:
        """Calculate precision and recall for autocorrect"""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for incorrect, expected in test_cases:
            correction = autocorrect.correct(incorrect)
            
            # True positive: incorrect word was corrected to expected word
            if incorrect != expected and correction == expected:
                true_positives += 1
            
            # False positive: correct word was changed incorrectly
            elif incorrect == expected and correction != expected:
                false_positives += 1
            
            # False negative: incorrect word was not corrected properly
            elif incorrect != expected and correction != expected:
                false_negatives += 1
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def generate_comparison_report(self, systems: Dict[str, Any], 
                                 test_cases: Dict[str, List]) -> Dict[str, Any]:
        """Generate comprehensive comparison report for multiple systems"""
        report = {}
        
        # Evaluate autocomplete systems
        if 'autocomplete' in systems and 'autocomplete' in test_cases:
            autocomplete_results = {}
            for name, system in systems['autocomplete'].items():
                results = self.evaluate_autocomplete(system, test_cases['autocomplete'])
                autocomplete_results[name] = results
            report['autocomplete'] = autocomplete_results
        
        # Evaluate autocorrect systems
        if 'autocorrect' in systems and 'autocorrect' in test_cases:
            autocorrect_results = {}
            for name, system in systems['autocorrect'].items():
                results = self.evaluate_autocorrect(system, test_cases['autocorrect'])
                autocorrect_results[name] = results
            report['autocorrect'] = autocorrect_results
        
        return report
    
    def print_detailed_results(self):
        """Print detailed evaluation results"""
        print("\n" + "="*60)
        print("NLP SYSTEMS EVALUATION RESULTS")
        print("="*60)
        
        for system_type, results in self.results.items():
            print(f"\n{system_type.upper()} SYSTEM:")
            print("-" * 40)
            
            if 'accuracy' in results:
                print(f"Accuracy: {results['accuracy']:.4f}")
            
            if 'response_time_ms' in results:
                rt = results['response_time_ms']
                print(f"Response Time: {rt['mean']:.2f}ms (±{rt['std']:.2f}ms)")
            
            if 'keystroke_savings' in results:
                ks = results['keystroke_savings']
                print(f"Keystroke Savings: {ks['savings_percentage']:.2f}%")
            
            if 'precision' in results and 'recall' in results:
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1-Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    # Test the evaluator
    evaluator = NLPEvaluator()
    
    # Sample test cases
    autocomplete_test_cases = [
        ("the quick", "brown"),
        ("machine", "learning"),
        ("natural language", "processing"),
        ("deep", "learning")
    ]
    
    autocorrect_test_cases = [
        ("quik", "quick"),
        ("browm", "brown"),
        ("mashine", "machine"),
        ("lurn", "learning"),
        ("procesing", "processing")
    ]
    
    # Create sample systems
    from .autocomplete import AutoComplete
    from .autocorrect import AutoCorrect
    
    sample_texts = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a subset of artificial intelligence",
        "natural language processing enables computers to understand human language"
    ]
    
    # Test autocomplete
    ac = AutoComplete()
    ac.train(sample_texts)
    ac_results = evaluator.evaluate_autocomplete(ac, autocomplete_test_cases)
    
    # Test autocorrect
    corrector = AutoCorrect()
    corrector.train(sample_texts)
    corrector_results = evaluator.evaluate_autocorrect(corrector, autocorrect_test_cases)
    
    # Print results
    evaluator.print_detailed_results()
    