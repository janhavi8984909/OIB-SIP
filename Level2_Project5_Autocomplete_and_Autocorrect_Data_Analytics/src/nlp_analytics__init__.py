"""
NLP Analytics Module
Autocomplete and Autocorrect System
"""

__version__ = "1.0.0"
__author__ = "NLP Team"

from .data_collection import DataCollector
from .text_preprocessing import TextPreprocessor
from .autocomplete import AutoComplete
from .autocorrect import AutoCorrect
from .metrics import NLPEvaluator

__all__ = [
    'DataCollector',
    'TextPreprocessor', 
    'AutoComplete',
    'AutoCorrect',
    'NLPEvaluator'
]
