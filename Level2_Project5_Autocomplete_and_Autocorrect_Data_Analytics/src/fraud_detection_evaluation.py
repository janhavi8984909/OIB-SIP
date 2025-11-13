import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionEvaluator:
    """Comprehensive evaluation for fraud detection models"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str = "model") -> Dict[str, float]:
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"{model_name} - AP: {metrics['average_precision']:.4f}, "
                   f"F1: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, models: Dict[str, Any], 
                          X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Evaluate multiple models and return comparison DataFrame"""
        all_metrics = {}
        
        for name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, name)
            all_metrics[name] = metrics
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics).T
        comparison_df = comparison_df.sort_values('average_precision', ascending=False)
        
        return comparison_df
    
    def plot_precision_recall_curve(self, models: Dict[str, Any], 
                                  X_test: pd.DataFrame, y_test: pd.Series):
        """Plot precision-recall curve for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            ap_score = average_precision_score(y_test, y_pred_proba)
            
            plt.plot(recall, precision, label=f'{name} (AP = {ap_score:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_confusion_matrix(self, model, X_test: pd.DataFrame, 
                            y_test: pd.Series, model_name: str = "Model"):
        """Plot confusion matrix for a single model"""
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Genuine', 'Fraud'],
                   yticklabels=['Genuine', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.show()
    
    def generate_report(self, models: Dict[str, Any], 
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {}
        
        # Model comparisons
        comparison_df = self.evaluate_all_models(models, X_test, y_test)
        report['model_comparison'] = comparison_df
        
        # Best model
        best_model_name = comparison_df.index[0]
        report['best_model'] = best_model_name
        report['best_model_metrics'] = comparison_df.loc[best_model_name].to_dict()
        
        # Detailed classification reports
        report['detailed_reports'] = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            report['detailed_reports'][name] = classification_report(
                y_test, y_pred, output_dict=True
            )
        
        return report
    