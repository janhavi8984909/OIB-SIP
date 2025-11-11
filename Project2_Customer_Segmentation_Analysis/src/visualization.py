"""
Visualization module for customer segmentation analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_elbow_method(cluster_results, save_path=None):
    """
    Plot elbow method and silhouette analysis.
    
    Parameters:
    -----------
    cluster_results : dict
        Results from find_optimal_clusters method
    save_path : str, optional
        Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow method plot
    wcss = cluster_results['wcss']
    ax1.plot(range(1, len(wcss) + 1), wcss, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores plot
    silhouette_scores = cluster_results['silhouette_scores']
    ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'ro-', 
             linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis for Optimal k')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Elbow method plot saved to {save_path}")
    
    plt.show()


def plot_cluster_analysis(df, save_path=None):
    """
    Plot cluster analysis results.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with cluster labels
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot: Income vs Total Spending
    scatter = axes[0, 0].scatter(df['Income'], df['MntTotal'], c=df['Cluster'], 
                                cmap='viridis', alpha=0.6, s=50)
    axes[0, 0].set_xlabel('Income')
    axes[0, 0].set_ylabel('Total Amount Spent')
    axes[0, 0].set_title('Customer Segments: Income vs Total Spending')
    plt.colorbar(scatter, ax=axes[0, 0], label='Cluster')
    
    # Box plot: Income by cluster
    sns.boxplot(data=df, x='Cluster', y='Income', ax=axes[0, 1])
    axes[0, 1].set_title('Income Distribution by Cluster')
    
    # Box plot: Total spending by cluster
    sns.boxplot(data=df, x='Cluster', y='MntTotal', ax=axes[1, 0])
    axes[1, 0].set_title('Total Spending Distribution by Cluster')
    
    # Cluster sizes
    cluster_sizes = df['Cluster'].value_counts().sort_index()
    axes[1, 1].bar(cluster_sizes.index, cluster_sizes.values, color='lightblue')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Cluster Sizes')
    
    # Add percentage labels on bars
    for i, (cluster, size) in enumerate(cluster_sizes.items()):
        percentage = (size / len(df)) * 100
        axes[1, 1].text(cluster, size, f'{percentage:.1f}%', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Cluster analysis plot saved to {save_path}")
    
    plt.show()


def plot_cluster_profiles(cluster_profiles, save_path=None):
    """
    Plot detailed cluster profiles.
    
    Parameters:
    -----------
    cluster_profiles : dict
        Cluster profiles from analysis
    save_path : str, optional
        Path to save the plot
    """
    clusters = list(cluster_profiles.keys())
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Average Income by Cluster
    incomes = [cluster_profiles[cluster]['demographics']['avg_income'] 
               for cluster in clusters]
    axes[0, 0].bar(clusters, incomes, color='skyblue')
    axes[0, 0].set_title('Average Income by Cluster')
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Average Income ($)')
    
    # Total Spending by Cluster
    spendings = [cluster_profiles[cluster]['spending_behavior']['total_spending'] 
                 for cluster in clusters]
    axes[0, 1].bar(clusters, spendings, color='lightcoral')
    axes[0, 1].set_title('Average Total Spending by Cluster')
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Average Spending ($)')
    
    # Relationship Rate by Cluster
    relationship_rates = [cluster_profiles[cluster]['demographics']['relationship_rate'] 
                         for cluster in clusters]
    axes[1, 0].bar(clusters, relationship_rates, color='lightgreen')
    axes[1, 0].set_title('Relationship Rate by Cluster')
    axes[1, 0].set_xlabel('Cluster')
    axes[1, 0].set_ylabel('Relationship Rate (%)')
    
    # Cluster Sizes
    sizes = [cluster_profiles[cluster]['size'] for cluster in clusters]
    percentages = [cluster_profiles[cluster]['percentage'] for cluster in clusters]
    bars = axes[1, 1].bar(clusters, sizes, color='gold')
    axes[1, 1].set_title('Cluster Sizes')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Customers')
    
    # Add percentage labels
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Cluster profiles plot saved to {save_path}")
    
    plt.show()
    