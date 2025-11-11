import pickle
import joblib
import json
import os
from datetime import datetime

class ModelSaver:
    def __init__(self, models_dir='models/trained_models/'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    def save_model(self, model, model_name, vectorizer=None):
        """Save trained model and vectorizer"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save vectorizer if provided
        if vectorizer is not None:
            vectorizer_filename = f"vectorizer_{model_name}_{timestamp}.pkl"
            vectorizer_path = os.path.join(self.models_dir, vectorizer_filename)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
        
        # Save model info
        model_info = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'vectorizer_path': vectorizer_path if vectorizer else None
        }
        
        return model_info
    
    def load_model(self, model_path, vectorizer_path=None):
        """Load saved model and vectorizer"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        vectorizer = None
        if vectorizer_path and os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
        
        return model, vectorizer
    
    def save_training_history(self, history, model_name):
        """Save training history for neural networks"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_filename = f"training_history_{model_name}_{timestamp}.json"
        history_path = os.path.join(self.models_dir, history_filename)
        
        with open(history_path, 'w') as f:
            json.dump(history, f)
        
        return history_path