import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing for fraud detection"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scalers = {}
        self.feature_columns = None
        
    def preprocess_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Main preprocessing pipeline"""
        logger.info("Starting data preprocessing...")
        
        # Separate features and target
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self._scale_features(X)
        
        # Handle class imbalance
        X_resampled, y_resampled = self._handle_imbalance(X_scaled, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_resampled
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns
        }
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        logger.info("Scaling features...")
        
        X_scaled = X.copy()
        
        # Use RobustScaler for Amount (less sensitive to outliers)
        if 'Amount' in X.columns:
            self.scalers['amount'] = RobustScaler()
            X_scaled['Amount'] = self.scalers['amount'].fit_transform(
                X[['Amount']]
            ).flatten()
        
        # Use StandardScaler for Time
        if 'Time' in X.columns:
            self.scalers['time'] = StandardScaler()
            X_scaled['Time'] = self.scalers['time'].fit_transform(
                X[['Time']]
            ).flatten()
        
        # Scale V1-V28 features (already PCA transformed, but standardize)
        v_columns = [col for col in X.columns if col.startswith('V')]
        if v_columns:
            self.scalers['v_features'] = StandardScaler()
            X_scaled[v_columns] = self.scalers['v_features'].fit_transform(
                X[v_columns]
            )
        
        return X_scaled
    
    def _handle_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE and undersampling"""
        logger.info("Handling class imbalance...")
        
        # First use SMOTE to generate synthetic fraud cases
        smote = SMOTE(
            sampling_strategy=0.1,  # Increase fraud to 10% of majority class
            random_state=self.random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Then undersample majority class
        undersampler = RandomUnderSampler(
            sampling_strategy=0.5,  # 2:1 ratio (genuine:fraud)
            random_state=self.random_state
        )
        X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
        
        logger.info(f"After resampling - Fraud rate: {y_resampled.mean():.4f}")
        
        return X_resampled, y_resampled
    
    def transform_new_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scalers"""
        if not self.scalers:
            raise ValueError("Preprocessor must be fitted first")
        
        X_transformed = X.copy()
        
        # Apply same scaling transformations
        if 'Amount' in X.columns and 'amount' in self.scalers:
            X_transformed['Amount'] = self.scalers['amount'].transform(
                X[['Amount']]
            ).flatten()
        
        if 'Time' in X.columns and 'time' in self.scalers:
            X_transformed['Time'] = self.scalers['time'].transform(
                X[['Time']]
            ).flatten()
        
        v_columns = [col for col in X.columns if col.startswith('V')]
        if v_columns and 'v_features' in self.scalers:
            X_transformed[v_columns] = self.scalers['v_features'].transform(
                X[v_columns]
            )
        
        return X_transformed
    