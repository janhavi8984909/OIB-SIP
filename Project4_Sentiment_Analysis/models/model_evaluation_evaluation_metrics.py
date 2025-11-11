import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred, y_prob=None, model_name='model'):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Class-wise metrics
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC if probabilities are available
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                metrics['roc_auc'] = None
        
        self.results[model_name] = {
            'metrics': metrics,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        
        return self.results[model_name]
    
    def generate_comparison_report(self, output_path='models/model_evaluation/model_comparison.json'):
        """Generate comparison report for all models"""
        comparison_data = {}
        
        for model_name, result in self.results.items():
            comparison_data[model_name] = {
                'accuracy': result['metrics']['accuracy'],
                'precision_macro': result['metrics']['precision_macro'],
                'recall_macro': result['metrics']['recall_macro'],
                'f1_macro': result['metrics']['f1_macro'],
                'roc_auc': result['metrics'].get('roc_auc', 'N/A')
            }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df = comparison_df.sort_values('f1_macro', ascending=False)
        
        return comparison_df
    
    def plot_model_comparison(self, metrics=['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']):
        """Plot comparison of models across different metrics"""
        if not self.results:
            print("No results to compare")
            return
        
        comparison_data = {}
        for model_name, result in self.results.items():
            comparison_data[model_name] = {metric: result['metrics'][metric] for metric in metrics}
        
        df = pd.DataFrame(comparison_data).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            df[metric].sort_values().plot(kind='barh', ax=axes[i], color='skyblue')
            axes[i].set_title(f'Model Comparison - {metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Score')
            
            # Add value labels
            for j, v in enumerate(df[metric].sort_values()):
                axes[i].text(v + 0.01, j, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
        
        return df