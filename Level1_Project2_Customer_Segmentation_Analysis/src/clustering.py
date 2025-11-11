"""
Clustering analysis module for customer segmentation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt


class CustomerSegmentation:
    """
    Customer segmentation using K-means clustering.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.labels = None
        
    def find_optimal_clusters(self, X, max_k=10):
        """
        Find optimal number of clusters using multiple methods.
        
        Parameters:
        -----------
        X : array-like
            Scaled feature matrix
        max_k : int
            Maximum number of clusters to test
            
        Returns:
        --------
        dict
            Results from different methods
        """
        results = {}
        
        # Elbow method - WCSS
        wcss = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        results['wcss'] = wcss
        
        # Silhouette scores (for k >= 2)
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
        results['silhouette_scores'] = silhouette_scores
        
        return results
    
    def perform_kmeans_clustering(self, X, n_clusters=4):
        """
        Perform K-means clustering.
        
        Parameters:
        -----------
        X : array-like
            Scaled feature matrix
        n_clusters : int
            Number of clusters
            
        Returns:
        --------
        tuple
            (labels, metrics)
        """
        self.model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        self.labels = self.model.fit_predict(X)
        
        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette_score(X, self.labels),
            'calinski_harabasz_score': calinski_harabasz_score(X, self.labels),
            'davies_bouldin_score': davies_bouldin_score(X, self.labels)
        }
        
        print(f"✅ K-means clustering completed with {n_clusters} clusters")
        print(f"   Silhouette Score: {metrics['silhouette_score']:.3f}")
        
        return self.labels, metrics
    
    def analyze_clusters(self, df, labels, feature_names):
        """
        Analyze cluster characteristics.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Original dataframe
        labels : array-like
            Cluster labels
        feature_names : list
            Names of features used in clustering
            
        Returns:
        --------
        tuple
            (cluster_summary, insights)
        """
        df_analysis = df.copy()
        df_analysis['Cluster'] = labels
        
        # Cluster characteristics
        cluster_summary = df_analysis.groupby('Cluster')[feature_names].mean()
        
        # Additional insights
        insights = {}
        for cluster in sorted(df_analysis['Cluster'].unique()):
            cluster_data = df_analysis[df_analysis['Cluster'] == cluster]
            insights[cluster] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(df_analysis) * 100,
                'avg_income': cluster_data['Income'].mean(),
                'avg_total_spending': cluster_data['MntTotal'].mean(),
                'relationship_rate': cluster_data['In_relationship'].mean() * 100
            }
        
        return cluster_summary, insights
    