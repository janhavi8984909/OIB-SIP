# src/models/base_model.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class BaseModel(ABC):
    def __init__(self, name, **params):
        self.name = name
        self.model = None
        self.params = params
        self.is_fitted = False
    
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    def fit(self, X, y, **kwargs):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        self.model.fit(X, y, **kwargs)
        self.is_fitted = True
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, threshold=0.5):
        """Evaluate model performance"""
        y_pred = (self.predict_proba(X)[:, 1] > threshold).astype(int)
        
        results = {
            'predictions': y_pred,
            'probabilities': self.predict_proba(X)[:, 1],
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return results
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
        else:
            raise ValueError("No model to save")
    
    def load(self, filepath):
        """Load model from file"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        