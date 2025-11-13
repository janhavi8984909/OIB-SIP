import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loader for credit card fraud detection dataset"""
    
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['paths']['raw_data'])
    
    def _load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def load_credit_card_data(self, file_name="creditcard.csv"):
        """Load credit card fraud detection dataset"""
        try:
            file_path = self.data_path / file_name
            logger.info(f"Loading data from {file_path}")
            
            data = pd.read_csv(file_path)
            logger.info(f"Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            logger.info(f"Fraud rate: {data['Class'].mean():.4f}")
            
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

def load_credit_card_data(file_name="creditcard.csv"):
    """Convenience function to load credit card data"""
    loader = DataLoader()
    return loader.load_credit_card_data(file_name)

if __name__ == "__main__":
    # Test the data loader
    data = load_credit_card_data()
    print(data.head())
    print(f"Dataset shape: {data.shape}")
    print(f"Fraud percentage: {data['Class'].mean() * 100:.2f}%")
    