# src/training/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import json
from pathlib import Path

from ..utils.config import config
from ..utils.helpers import save_model, save_results
from ..models.logistic_regression import LogisticRegressionModel
from ..models.random_forest import RandomForestModel
from ..models.neural_network import NeuralNetworkModel

class ModelTrainer:
    def __init__(self):
        self.config = config.load_model_config()
        self.paths = config.get_paths()
        self.models = {}
    
    def load_data(self):
        """Load processed training data"""
        train_path = self.paths['data_processed'] / "train_data.csv"
        test_path = self.paths['data_processed'] / "test_data.csv"
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        X_train = train_data.drop('Class', axis=1)
        y_train = train_data['Class']
        X_test = test_data.drop('Class', axis=1)
        y_test = test_data['Class']
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize all models from configuration"""
        model_configs = self.config['models']
        
        self.models['logistic_regression'] = LogisticRegressionModel(
            **model_configs['logistic_regression']['params']
        )
        
        self.models['random_forest'] = RandomForestModel(
            **model_configs['random_forest']['params']
        )
        
        self.models['neural_network'] = NeuralNetworkModel(
            **model_configs['neural_network']['params']
        )
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Save trained model
            model_path = self.paths['models'] / f"{name}.pkl"
            model.save(model_path)
            
            results[name] = {
                'status': 'trained',
                'model_path': str(model_path)
            }
        
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        evaluation_results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import (
                precision_score, recall_score, f1_score, 
                roc_auc_score, average_precision_score,
                confusion_matrix, classification_report
            )
            
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            auprc = auc(recall, precision)
            
            results = {
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'average_precision': average_precision_score(y_test, y_pred_proba),
                'auprc': auprc,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            evaluation_results[name] = results
        
        return evaluation_results
    
    def run_training_pipeline(self):
        """Run complete training pipeline"""
        print("Starting model training pipeline...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        # Initialize models
        self.initialize_models()
        
        # Train models
        training_results = self.train_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # Save results
        save_results(training_results, "training_results.json")
        save_results(evaluation_results, "evaluation_results.json")
        
        # Print best model
        best_model = max(evaluation_results.items(), 
                        key=lambda x: x[1]['average_precision'])
        print(f"\nBest model: {best_model[0]} with AP = {best_model[1]['average_precision']:.4f}")
        
        return training_results, evaluation_results
    