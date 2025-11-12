# src/utils/config.py
import yaml
import os
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.config_dir = self.base_dir / "configuration"
        
    def load_model_config(self):
        """Load model configuration"""
        with open(self.config_dir / "model_config.yaml", 'r') as file:
            return yaml.safe_load(file)
    
    def load_preprocessing_config(self):
        """Load preprocessing configuration"""
        with open(self.config_dir / "preprocessing_config.yaml", 'r') as file:
            return yaml.safe_load(file)
    
    def load_monitoring_config(self):
        """Load monitoring configuration"""
        with open(self.config_dir / "monitoring_config.yaml", 'r') as file:
            return yaml.safe_load(file)
    
    def get_paths(self):
        """Get all project paths"""
        return {
            'data_raw': self.base_dir / "data" / "raw",
            'data_processed': self.base_dir / "data" / "processed",
            'models': self.base_dir / "models",
            'results': self.base_dir / "results",
            'notebooks': self.base_dir / "notebooks"
        }

# Singleton instance
config = Config()
