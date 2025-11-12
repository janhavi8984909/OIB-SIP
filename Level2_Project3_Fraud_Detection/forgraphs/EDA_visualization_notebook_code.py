# notebooks/02_eda_visualization.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AdvancedEDAVisualizer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.results_dir = Path("results/figures")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load processed data"""
        try:
            df = pd.read_csv(self.data_path / "processed" / "cleaned_data.csv")
        except:
            df = pd.read_csv(self.data_path / "raw" / "creditcard.csv")
        return df
    
    def plot_feature_importance_analysis(self, df):
        """Plot feature importance using various methods"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import mutual_info_classif
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot Random Forest importance
        ax1.barh(rf_importance['feature'][:15], rf_importance['importance'][:15], 
                color='teal', alpha=0.7)
        ax1.set_title('Top 15 Features - Random Forest Importance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Importance Score')
        
        # Plot Mutual Information importance
        ax2.barh(mi_importance['feature'][:15], mi_importance['importance'][:15], 
                color='purple', alpha=0.7)
        ax2.set_title('Top 15 Features - Mutual Information', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return rf_importance, mi_importance
    
    def plot_tsne_visualization(self, df, sample_size=5000):
        """Plot t-SNE visualization of the data"""
        # Sample data for t-SNE (it's computationally expensive)
        sample_df = df.sample(min(sample_size, len(df)), random_state=42)
        
        X = sample_df.drop('Class', axis=1)
        y = sample_df['Class']
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                            alpha=0.6, s=30)
        plt.colorbar(scatter, label='Class (0: Legitimate, 1: Fraud)')
        plt.title('t-SNE Visualization of Credit Card Transactions', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='yellow', markersize=8, label='Legitimate'),
                         plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='purple', markersize=8, label='Fraud')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "tsne_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, df):
        """Plot PCA analysis results"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Scree plot
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance, 
               alpha=0.6, color='blue', label='Individual')
        ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                'ro-', label='Cumulative')
        ax1.set_xlabel('Principal Components')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Scree Plot', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PCA visualization (first two components)
        scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', 
                            alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        ax2.set_title('PCA - First Two Components', fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Class')
        ax2.grid(True, alpha=0.3)
        
        # Feature loadings for first two components
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        
        # Plot top feature contributions to first two components
        n_top_features = 10
        feature_names = X.columns
        
        # PC1 contributions
        pc1_contributions = pd.DataFrame({
            'feature': feature_names,
            'contribution': np.abs(loadings[:, 0])
        }).sort_values('contribution', ascending=False).head(n_top_features)
        
        ax3.barh(pc1_contributions['feature'], pc1_contributions['contribution'], 
                color='green', alpha=0.7)
        ax3.set_title(f'Top {n_top_features} Features - PC1 Contribution', fontweight='bold')
        ax3.set_xlabel('Absolute Contribution')
        
        # PC2 contributions
        pc2_contributions = pd.DataFrame({
            'feature': feature_names,
            'contribution': np.abs(loadings[:, 1])
        }).sort_values('contribution', ascending=False).head(n_top_features)
        
        ax4.barh(pc2_contributions['feature'], pc2_contributions['contribution'], 
                color='orange', alpha=0.7)
        ax4.set_title(f'Top {n_top_features} Features - PC2 Contribution', fontweight='bold')
        ax4.set_xlabel('Absolute Contribution')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "pca_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Variance explained by first 2 PCs: {cumulative_variance[1]:.2%}")
        print(f"Variance explained by first 5 PCs: {cumulative_variance[4]:.2%}")
    
    def plot_interactive_distributions(self, df):
        """Create interactive distribution plots using Plotly"""
        # Amount distribution by class
        fig = px.histogram(df, x='Amount', color='Class', 
                          title='Transaction Amount Distribution by Class',
                          nbins=50, opacity=0.7,
                          color_discrete_map={0: 'blue', 1: 'red'},
                          labels={'Class': 'Transaction Type'})
        
        fig.update_layout(
            xaxis_title='Transaction Amount',
            yaxis_title='Count',
            legend_title='Transaction Type',
            font=dict(size=12)
        )
        
        fig.write_html(str(self.results_dir / "interactive_amount_distribution.html"))
        
        # Time distribution by class
        df['Hour'] = df['Time'] / 3600
        fig2 = px.histogram(df, x='Hour', color='Class',
                           title='Transaction Time Distribution by Class',
                           nbins=48, opacity=0.7,
                           color_discrete_map={0: 'green', 1: 'red'},
                           labels={'Class': 'Transaction Type'})
        
        fig2.update_layout(
            xaxis_title='Time (Hours)',
            yaxis_title='Count',
            legend_title='Transaction Type',
            font=dict(size=12)
        )
        
        fig2.write_html(str(self.results_dir / "interactive_time_distribution.html"))
        
        return fig, fig2

# Usage
if __name__ == "__main__":
    eda_visualizer = AdvancedEDAVisualizer("data")
    df = eda_visualizer.load_data()
    
    # Generate advanced EDA plots
    rf_imp, mi_imp = eda_visualizer.plot_feature_importance_analysis(df)
    eda_visualizer.plot_tsne_visualization(df)
    eda_visualizer.plot_pca_analysis(df)
    eda_visualizer.plot_interactive_distributions(df)
    