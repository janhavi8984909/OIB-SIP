"""
Utility functions for customer segmentation analysis.
"""

import pandas as pd
import numpy as np
import os


def create_directories():
    """
    Create necessary directories for the project.
    """
    directories = [
        'data/raw',
        'data/processed',
        'reports/figures/eda_visualizations',
        'reports/figures/clustering_results',
        'reports/figures/feature_importance',
        'models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def save_results(df, cluster_summary, output_dir='data/processed'):
    """
    Save analysis results to CSV files.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels
    cluster_summary : pd.DataFrame
        Cluster summary statistics
    output_dir : str
        Output directory path
    """
    # Save clustered data
    df_output_path = os.path.join(output_dir, 'customer_segmentation_with_clusters.csv')
    df.to_csv(df_output_path, index=False)
    print(f"💾 Clustered data saved to: {df_output_path}")
    
    # Save cluster summary
    summary_output_path = os.path.join(output_dir, 'cluster_summary.csv')
    cluster_summary.to_csv(summary_output_path)
    print(f"💾 Cluster summary saved to: {summary_output_path}")


def print_cluster_insights(insights):
    """
    Print formatted cluster insights.
    
    Parameters:
    -----------
    insights : dict
        Cluster insights from analysis
    """
    print("\n" + "="*60)
    print("📊 CLUSTER INSIGHTS")
    print("="*60)
    
    for cluster, info in insights.items():
        print(f"\n🔹 Cluster {cluster}:")
        print(f"   • Size: {info['size']} customers ({info['percentage']:.1f}%)")
        print(f"   • Average Income: ${info['avg_income']:,.2f}")
        print(f"   • Average Total Spending: ${info['avg_total_spending']:,.2f}")
        print(f"   • Relationship Rate: {info['relationship_rate']:.1f}%")


def validate_data(df):
    """
    Validate the dataset for analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to validate
        
    Returns:
    --------
    bool
        True if validation passes, False otherwise
    """
    required_columns = ['Income', 'MntTotal', 'Kidhome', 'Teenhome', 'Recency']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    if df.empty:
        print("❌ Dataset is empty")
        return False
    
    print("✅ Data validation passed")
    return True
