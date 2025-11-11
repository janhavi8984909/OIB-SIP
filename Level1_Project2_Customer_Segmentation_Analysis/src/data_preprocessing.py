"""
Data preprocessing module for customer segmentation analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load the customer segmentation dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {file_path}")
        return None


def clean_data(df):
    """
    Clean the dataset by handling missing values and inconsistencies.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset
    """
    df_clean = df.copy()
    
    # Handle missing values in numeric columns
    numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        df_clean[numeric_columns] = numeric_imputer.fit_transform(df_clean[numeric_columns])
    
    # Remove duplicates
    initial_rows = df_clean.shape[0]
    df_clean = df_clean.drop_duplicates()
    final_rows = df_clean.shape[0]
    
    if initial_rows != final_rows:
        print(f"📝 Removed {initial_rows - final_rows} duplicate rows")
    
    print("✅ Data cleaning completed")
    return df_clean


def engineer_features(df):
    """
    Create new features for better segmentation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset with engineered features
    """
    df_eng = df.copy()
    
    # Relationship status
    df_eng['In_relationship'] = df_eng['marital_Married'] + df_eng['marital_Together']
    df_eng['In_relationship'] = df_eng['In_relationship'].apply(lambda x: 1 if x >= 1 else 0)
    
    # Total children
    df_eng['Total_Children'] = df_eng['Kidhome'] + df_eng['Teenhome']
    
    # Total purchases across all channels
    df_eng['Total_Purchases'] = (
        df_eng['NumDealsPurchases'] + 
        df_eng['NumCatalogPurchases'] + 
        df_eng['NumStorePurchases'] + 
        df_eng['NumWebPurchases']
    )
    
    # Customer value score
    df_eng['Customer_Value_Score'] = (
        df_eng['MntTotal'] * 0.5 + 
        (1 / (df_eng['Recency'] + 1)) * 0.3 + 
        df_eng['Total_Purchases'] * 0.2
    )
    
    print("✅ Feature engineering completed")
    return df_eng


def prepare_clustering_data(df, features=None):
    """
    Prepare data for clustering analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with engineered features
    features : list, optional
        List of features to use for clustering
        
    Returns:
    --------
    tuple
        (X_scaled, feature_names, scaler)
    """
    if features is None:
        features = [
            'Income', 'MntTotal', 'In_relationship', 'Total_Children',
            'NumWebPurchases', 'NumStorePurchases', 'Recency', 'Customer_Value_Score'
        ]
    
    # Select features
    X = df[features].copy()
    
    # Handle any remaining missing values
    X = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✅ Data prepared for clustering: {X_scaled.shape}")
    return X_scaled, features, scaler
