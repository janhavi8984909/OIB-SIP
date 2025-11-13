import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector:
    """Fraud detection model trainer and predictor"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.is_trained = False
        
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train multiple fraud detection models"""
        logger.info("Training fraud detection models...")
        
        # Define models to train
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced',
                max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                class_weight='balanced',
                n_estimators=100
            ),
            'xgboost': XGBClassifier(
                random_state=self.random_state,
                scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        # Train each model
        trained_models = {}
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        self.models = trained_models
        self.is_trained = True
        
        logger.info("All models trained successfully")
        return trained_models
    
    def predict(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Make predictions using trained model"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name].predict(X)
        else:
            # Use best model if available, otherwise use random forest
            model = self.best_model if self.best_model else self.models['random_forest']
            return model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            return self.models[model_name].predict_proba(X)
        else:
            model = self.best_model if self.best_model else self.models['random_forest']
            return model.predict_proba(X)
    
    def set_best_model(self, model_name: str):
        """Set the best performing model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        self.best_model = self.models[model_name]
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        if not self.is_trained:
            raise ValueError("No trained models to save")
        
        for name, model in self.models.items():
            model_path = f"{path}/{name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} to {model_path}")
    
    def load_models(self, path: str, model_names: List[str]):
        """Load trained models from disk"""
        loaded_models = {}
        for name in model_names:
            model_path = f"{path}/{name}_model.pkl"
            try:
                model = joblib.load(model_path)
                loaded_models[name] = model
                logger.info(f"Loaded {name} from {model_path}")
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
        
        self.models = loaded_models
        self.is_trained = bool(loaded_models)

class AnomalyDetector:
    """Anomaly detection-based fraud detection"""
    
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.is_trained = False
    
    def train(self, X: pd.DataFrame):
        """Train isolation forest for anomaly detection"""
        logger.info("Training Isolation Forest for anomaly detection...")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        self.model.fit(X)
        self.is_trained = True
        logger.info("Anomaly detection model trained")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(X)
        # Convert to binary (0 for normal, 1 for fraud/anomaly)
        return (predictions == -1).astype(int)
    