# src/models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self, **params):
        super().__init__("random_forest", **params)
    
    def build_model(self):
        self.model = RandomForestClassifier(**self.params)
        