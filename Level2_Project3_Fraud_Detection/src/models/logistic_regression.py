# src/models/logistic_regression.py
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self, **params):
        super().__init__("logistic_regression", **params)
    
    def build_model(self):
        self.model = LogisticRegression(**self.params)
        