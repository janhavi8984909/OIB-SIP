import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def setup_plot_style():
    """Set up consistent plotting style"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_feature_importance(feature_importance: Dict[str, float], 
                          top_n: int = 15, 
                          title: str = "Feature Importance"):
    """Plot feature importance for machine learning models"""
    setup_plot_style()
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importance = zip(*sorted_features)
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, importance, align='center', alpha=0.7)
    plt.yticks(y_pos, features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, 
                        y_pred: np.ndarray, 
                        labels: List[str] = None,
                        title: str = "Confusion Matrix"):
    """Plot confusion matrix"""
    setup_plot_style()
    
    if labels is None:
        labels = ['Genuine', 'Fraud']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(y_true: np.ndarray, 
                              y_scores: np.ndarray, 
                              model_name: str = "Model"):
    """Plot precision-recall curve"""
    setup_plot_style()
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = np.trapz(precision, recall)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2, 
             label=f'{model_name} (AP = {average_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true: np.ndarray, 
                  y_scores: np.ndarray, 
                  model_name: str = "Model"):
    """Plot ROC curve"""
    setup_plot_style()
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, 
             label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_ngram_frequency(ngram_counts: Dict[str, int], 
                        top_n: int = 20, 
                        title: str = "Most Common N-grams"):
    """Plot frequency of most common n-grams"""
    setup_plot_style()
    
    sorted_ngrams = sorted(ngram_counts.items(), 
                         key=lambda x: x[1], reverse=True)[:top_n]
    
    ngrams, counts = zip(*sorted_ngrams)
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(ngrams))
    
    plt.barh(y_pos, counts, align='center', alpha=0.7)
    plt.yticks(y_pos, ngrams)
    plt.xlabel('Frequency')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_response_times(response_times: List[float], 
                      system_name: str = "System"):
    """Plot response time distribution"""
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(response_times, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Response Time (ms)')
    plt.ylabel('Frequency')
    plt.title(f'{system_name} - Response Time Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(response_times)
    plt.ylabel('Response Time (ms)')
    plt.title(f'{system_name} - Response Time Box Plot')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(comparison_df: pd.DataFrame, 
                        metric: str = 'average_precision'):
    """Plot model comparison for a specific metric"""
    setup_plot_style()
    
    plt.figure(figsize=(10, 6))
    
    models = comparison_df.index
    scores = comparison_df[metric]
    
    y_pos = np.arange(len(models))
    
    plt.barh(y_pos, scores, align='center', alpha=0.7)
    plt.yticks(y_pos, models)
    plt.xlabel(metric.replace('_', ' ').title())
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
    
    # Add value labels
    for i, v in enumerate(scores):
        plt.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def create_evaluation_dashboard(fraud_results: Dict[str, Any], 
                              nlp_results: Dict[str, Any]):
    """Create a comprehensive evaluation dashboard"""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Fraud detection metrics
    if 'model_comparison' in fraud_results:
        comparison_df = fraud_results['model_comparison']
        models = comparison_df.index
        ap_scores = comparison_df['average_precision']
        
        axes[0, 0].barh(models, ap_scores, alpha=0.7)
        axes[0, 0].set_title('Fraud Detection - Average Precision')
        axes[0, 0].set_xlabel('Average Precision')
    
    # NLP accuracy comparison
    if 'autocomplete' in nlp_results:
        ac_results = nlp_results['autocomplete']
        systems = list(ac_results.keys())
        accuracies = [results['accuracy'] for results in ac_results.values()]
        
        axes[0, 1].bar(systems, accuracies, alpha=0.7)
        axes[0, 1].set_title('Autocomplete - Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
    
    # Response times
    if 'autocorrect' in nlp_results:
        corrector_results = nlp_results['autocorrect']
        systems = list(corrector_results.keys())
        response_times = [results['response_time_ms']['mean'] 
                         for results in corrector_results.values()]
        
        axes[1, 0].bar(systems, response_times, alpha=0.7)
        axes[1, 0].set_title('Autocorrect - Response Time')
        axes[1, 0].set_ylabel('Response Time (ms)')
    
    # F1 scores
    if 'autocorrect' in nlp_results:
        corrector_results = nlp_results['autocorrect']
        systems = list(corrector_results.keys())
        f1_scores = [results['f1_score'] for results in corrector_results.values()]
        
        axes[1, 1].bar(systems, f1_scores, alpha=0.7)
        axes[1, 1].set_title('Autocorrect - F1 Score')
        axes[1, 1].set_ylabel('F1 Score')
    
    plt.tight_layout()
    plt.show()
    