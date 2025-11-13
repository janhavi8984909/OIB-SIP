
## 2. Source Code Files

### `03_src/fraud_detection/__init__.py`
```python
"""
Fraud Detection Module
Credit Card Fraud Detection System
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

from .data_loader import load_credit_card_data
from .preprocessing import DataPreprocessor
from .models import FraudDetector
from .evaluation import FraudDetectionEvaluator

__all__ = [
    'load_credit_card_data',
    'DataPreprocessor', 
    'FraudDetector',
    'FraudDetectionEvaluator'
]
