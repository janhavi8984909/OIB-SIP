# src/utils/helpers.py
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path

def save_model(model, filename, model_dir="models"):
    """Save trained model to file"""
    model_path = Path(model_dir) / filename
    model_path.parent.mkdir(exist_ok=True)
    
    if filename.endswith('.pkl'):
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
    elif filename.endswith('.h5'):
        model.save(model_path)
    
    print(f"Model saved to: {model_path}")

def load_model(filename, model_dir="models"):
    """Load trained model from file"""
    model_path = Path(model_dir) / filename
    
    if filename.endswith('.pkl'):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    elif filename.endswith('.h5'):
        from tensorflow.keras.models import load_model
        return load_model(model_path)

def save_results(results, filename, results_dir="results"):
    """Save evaluation results to JSON"""
    results_path = Path(results_dir) / filename
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=2)
    
    print(f"Results saved to: {results_path}")

def calculate_class_weights(y):
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))
