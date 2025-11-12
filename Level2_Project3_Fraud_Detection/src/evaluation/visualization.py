# src/evaluation/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, roc_curve, 
                           precision_recall_curve, auc)
from pathlib import Path

class ResultsVisualizer:
    def __init__(self, results_dir="results/figures"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, save=True):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save:
            filepath = self.results_dir / "confusion_matrices" / f"{model_name}_cm.png"
            filepath.parent.mkdir(exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, results_dict, save=True):
        """Plot ROC curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=self.colors[i % len(self.colors)], 
                    lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = self.results_dir / "roc_curves" / "model_comparison_roc.png"
            filepath.parent.mkdir(exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, results_dict, save=True):
        """Plot Precision-Recall curves for multiple models"""
        plt.figure(figsize=(10, 8))
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            y_true = results['y_true']
            y_pred_proba = results['y_pred_proba']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            avg_precision = auc(recall, precision)
            
            plt.plot(recall, precision, color=self.colors[i % len(self.colors)], 
                    lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        if save:
            filepath = self.results_dir / "roc_curves" / "model_comparison_pr.png"
            filepath.parent.mkdir(exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight")
        
        plt.show()
        