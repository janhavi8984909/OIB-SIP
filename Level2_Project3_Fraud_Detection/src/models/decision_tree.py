# src/models/decision_tree.py
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self, **params):
        super().__init__("decision_tree", **params)
    
    def build_model(self):
        self.model = DecisionTreeClassifier(**self.params)
        