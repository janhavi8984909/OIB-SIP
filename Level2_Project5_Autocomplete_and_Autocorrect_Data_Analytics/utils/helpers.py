import pickle
import joblib
import yaml
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import functools

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """Set up logging configuration"""
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': level,
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': level,
                'propagate': True
            }
        }
    }
    
    if log_file:
        logging_config['handlers']['file'] = {
            'level': level,
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': log_file,
            'mode': 'a'
        }
        logging_config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(logging_config)

def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def save_model(model: Any, file_path: str, method: str = 'joblib'):
    """Save a model to disk"""
    ensure_directory(Path(file_path).parent)
    
    try:
        if method == 'joblib':
            joblib.dump(model, file_path)
        elif method == 'pickle':
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported save method: {method}")
        
        logging.info(f"Model saved to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving model to {file_path}: {str(e)}")
        return False

def load_model(file_path: str, method: str = 'joblib') -> Any:
    """Load a model from disk"""
    try:
        if method == 'joblib':
            model = joblib.load(file_path)
        elif method == 'pickle':
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported load method: {method}")
        
        logging.info(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {file_path}: {str(e)}")
        raise

def timer(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logging.info(f"Function {func.__name__} executed in {execution_time:.4f} seconds")
        return result
    return wrapper

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise

def save_results(results: Dict[str, Any], file_path: str):
    """Save results to JSON file"""
    ensure_directory(Path(file_path).parent)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to {file_path}")
        return True
    except Exception as e:
        logging.error(f"Error saving results to {file_path}: {str(e)}")
        return False

def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        logging.info(f"Results loaded from {file_path}")
        return results
    except Exception as e:
        logging.error(f"Error loading results from {file_path}: {str(e)}")
        raise

def memory_usage():
    """Get current memory usage (cross-platform)"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        logging.warning("psutil not installed, memory usage not available")
        return None

class MemoryMonitor:
    """Context manager to monitor memory usage"""
    
    def __enter__(self):
        self.start_memory = memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = memory_usage()
        if self.start_memory and self.end_memory:
            memory_used = self.end_memory - self.start_memory
            logging.info(f"Memory used: {memory_used:.2f} MB")

def validate_dataframe(df, required_columns=None, allow_null_columns=None):
    """Validate DataFrame structure and data quality"""
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if allow_null_columns:
        null_counts = null_counts[~null_counts.index.isin(allow_null_columns)]
    
    if null_counts.any():
        problematic_columns = null_counts[null_counts > 0].index.tolist()
        logging.warning(f"Null values found in columns: {problematic_columns}")
    
    return True

def batch_process(data, batch_size=1000, process_func=None):
    """Process data in batches"""
    results = []
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        if process_func:
            batch_result = process_func(batch)
            results.extend(batch_result)
        
        if (i // batch_size) % 10 == 0:  # Log every 10 batches
            logging.info(f"Processed {i + len(batch)}/{len(data)} items")
    
    return results
