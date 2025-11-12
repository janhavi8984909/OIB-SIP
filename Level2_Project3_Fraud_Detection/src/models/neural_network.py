# src/models/neural_network.py
from sklearn.neural_network import MLPClassifier
from .base_model import BaseModel

class NeuralNetworkModel(BaseModel):
    def __init__(self, **params):
        super().__init__("neural_network", **params)
    
    def build_model(self):
        self.model = MLPClassifier(**self.params)
        