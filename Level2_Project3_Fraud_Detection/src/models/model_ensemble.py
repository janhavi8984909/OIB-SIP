# src/models/model_ensemble.py
import numpy as np
from sklearn.ensemble import VotingClassifier
from .base_model import BaseModel

class EnsembleModel(BaseModel):
    def __init__(self, models, voting='soft', weights=None):
        super().__init__("ensemble")
        self.models = models
        self.voting = voting
        self.weights = weights
    
    def build_model(self):
        estimators = [(model.name, model.model) for model in self.models]
        self.model = VotingClassifier(
            estimators=estimators,
            voting=self.voting,
            weights=self.weights
        )
    
    def fit(self, X, y, **kwargs):
        # First fit individual models
        for model in self.models:
            if not model.is_fitted:
                model.fit(X, y, **kwargs)
        
        # Then build and fit ensemble
        self.build_model()
        super().fit(X, y, **kwargs)
        