"""
Customer Segmentation Analysis Package

A comprehensive package for analyzing customer data and performing segmentation
using machine learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

from .data_preprocessing import load_data, clean_data, engineer_features
from .clustering import CustomerSegmentation
from .visualization import plot_elbow_method, plot_cluster_analysis

__all__ = [
    'load_data',
    'clean_data', 
    'engineer_features',
    'CustomerSegmentation',
    'plot_elbow_method',
    'plot_cluster_analysis'
]