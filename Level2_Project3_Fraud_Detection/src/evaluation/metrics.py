# src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionMetrics:
    def __init__(self):
        self.metrics_history = {}
    
    def calculate_business_metrics(self, y_true, y_pred, transaction_amounts=None):
        """Calculate business-specific metrics for fraud detection"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
        }
        
        # Calculate cost savings if transaction amounts are provided
        if transaction_amounts is not None:
            fraud_amounts = transaction_amounts[y_true == 1]
            detected_fraud_amounts = transaction_amounts[(y_true == 1) & (y_pred == 1)]
            metrics['fraud_amount_detected'] = detected_fraud_amounts.sum()
            metrics['fraud_amount_missed'] = fraud_amounts.sum() - detected_fraud_amounts.sum()
            metrics['fraud_detection_rate'] = detected_fraud_amounts.sum() / fraud_amounts.sum()
        
        return metrics
    
    def find_optimal_threshold(self, y_true, y_pred_proba, method='f1'):
        """Find optimal threshold based on different criteria"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        
        if method == 'f1':
            f1_scores = 2 * (precision * recall) / (precision + recall)
            optimal_idx = np.argmax(f1_scores[:-1])  # Exclude last element
            optimal_threshold = thresholds[optimal_idx]
        elif method == 'youden':
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = roc_thresholds[optimal_idx]
        elif method == 'cost':
            # Custom cost function (adjust weights based on business needs)
            cost_per_fp = 1  # Cost of false positive (customer inconvenience)
            cost_per_fn = 10  # Cost of false negative (fraud loss)
            costs = fp * cost_per_fp + fn * cost_per_fn
            optimal_idx = np.argmin(costs)
            optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold
    