# src/evaluation/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (roc_curve, precision_recall_curve, 
                           confusion_matrix, auc)

class ModelEvaluationVisualizer:
    def __init__(self, results_dir="results/figures"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3E92CC']
        
    def load_evaluation_results(self):
        """Load model evaluation results"""
        results_path = Path("results") / "evaluation_results.json"
        with open(results_path, 'r') as f:
            return json.load(f)
    
    def plot_model_comparison(self, results_dict):
        """Plot comprehensive model comparison"""
        metrics = ['precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Average Precision']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        models = list(results_dict.keys())
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            if i < len(axes) - 1:
                values = [results_dict[model][metric] for model in models]
                
                bars = axes[i].bar(models, values, color=self.colors[:len(models)], 
                                 alpha=0.8, edgecolor='black')
                axes[i].set_title(f'{metric_name} Comparison', fontweight='bold')
                axes[i].set_ylabel(metric_name)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
                
                axes[i].set_ylim(0, min(1.0, max(values) * 1.2))
                axes[i].grid(True, alpha=0.3, axis='y')
        
        # Create summary table in the last subplot
        axes[-1].axis('off')
        summary_data = []
        for model in models:
            row = [model]
            for metric in metrics:
                row.append(f"{results_dict[model][metric]:.3f}")
            summary_data.append(row)
        
        table = axes[-1].table(cellText=summary_data,
                             colLabels=['Model'] + metric_names,
                             cellLoc='center',
                             loc='center',
                             bbox=[0.1, 0.1, 0.9, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[-1].set_title('Model Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, results_dict, save=True):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            # For demonstration - in real scenario, you'd have true labels and probabilities
            # This is a placeholder - you'd need to store these during evaluation
            fpr = np.linspace(0, 1, 100)
            tpr = fpr ** (1/(i+1))  # Placeholder curve
            roc_auc = results.get('roc_auc', 0.8 + i*0.05)
            
            plt.plot(fpr, tpr, color=self.colors[i % len(self.colors)], 
                    lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save:
            roc_dir = self.results_dir / "roc_curves"
            roc_dir.mkdir(exist_ok=True)
            plt.savefig(roc_dir / "model_comparison_roc.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curves(self, results_dict, save=True):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            # Placeholder data - replace with actual precision-recall data
            recall = np.linspace(0, 1, 100)
            precision = np.exp(-recall * (i+1))  # Placeholder curve
            avg_precision = results.get('average_precision', 0.7 + i*0.05)
            
            plt.plot(recall, precision, color=self.colors[i % len(self.colors)], 
                    lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        
        if save:
            pr_dir = self.results_dir / "roc_curves"
            pr_dir.mkdir(exist_ok=True)
            plt.savefig(pr_dir / "model_comparison_pr.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(self, results_dict, save=True):
        """Plot confusion matrices for all models"""
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            cm = np.array(results.get('confusion_matrix', [[100, 10], [5, 2]]))
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['Predicted Legit', 'Predicted Fraud'],
                       yticklabels=['Actual Legit', 'Actual Fraud'])
            axes[i].set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            cm_dir = self.results_dir / "confusion_matrices"
            cm_dir.mkdir(exist_ok=True)
            plt.savefig(cm_dir / "model_comparison_cm.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_threshold_analysis(self, y_true, y_pred_proba, model_name):
        """Plot threshold analysis for a single model"""
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
        
        # Mark optimal threshold (max F1)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
                   label=f'Optimal Threshold: {optimal_threshold:.3f}')
        plt.plot(optimal_threshold, f1_scores[optimal_idx], 'ko', markersize=8)
        
        plt.xlabel('Classification Threshold')
        plt.ylabel('Score')
        plt.title(f'Threshold Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        plt.savefig(self.results_dir / f"threshold_analysis_{model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_threshold

# Usage example
if __name__ == "__main__":
    visualizer = ModelEvaluationVisualizer()
    
    # Load results (you would generate these from your model training)
    results = visualizer.load_evaluation_results()
    
    # Generate all evaluation plots
    visualizer.plot_model_comparison(results)
    visualizer.plot_roc_curves(results)
    visualizer.plot_precision_recall_curves(results)
    visualizer.plot_confusion_matrices(results)
    