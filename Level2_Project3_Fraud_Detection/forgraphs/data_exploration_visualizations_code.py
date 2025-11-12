# notebooks/01_data_exploration.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataExplorationVisualizer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.results_dir = Path("results/figures")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the credit card fraud dataset"""
        df = pd.read_csv(self.data_path / "raw" / "creditcard.csv")
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")
        return df
    
    def plot_class_distribution(self, df):
        """Plot class distribution with detailed statistics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        class_counts = df['Class'].value_counts()
        colors = ['#2E86AB', '#A23B72']
        
        ax1.bar(['Legitimate (0)', 'Fraud (1)'], class_counts.values, 
                color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Class Distribution - Count', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Transactions')
        
        # Add count labels on bars
        for i, count in enumerate(class_counts.values):
            ax1.text(i, count + 1000, f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        labels = ['Legitimate', 'Fraud']
        sizes = class_counts.values
        explode = (0, 0.1)
        
        ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.4f%%',
                shadow=True, startangle=90)
        ax2.set_title('Class Distribution - Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print(f"Legitimate transactions: {class_counts[0]:,} ({class_counts[0]/len(df)*100:.4f}%)")
        print(f"Fraudulent transactions: {class_counts[1]:,} ({class_counts[1]/len(df)*100:.4f}%)")
        print(f"Imbalance ratio: {class_counts[0]/class_counts[1]:.1f}:1")
    
    def plot_time_distribution(self, df):
        """Plot transaction distribution over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Convert time to hours
        df['Hour'] = df['Time'] / 3600
        
        # All transactions over time
        ax1.hist(df['Hour'], bins=48, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Time (Hours)')
        ax1.set_ylabel('Number of Transactions')
        ax1.set_title('Transaction Distribution Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Fraud transactions over time
        fraud_df = df[df['Class'] == 1]
        ax2.hist(fraud_df['Hour'], bins=48, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Time (Hours)')
        ax2.set_ylabel('Number of Fraud Transactions')
        ax2.set_title('Fraud Transaction Distribution Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "time_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_amount_distribution(self, df):
        """Plot transaction amount distribution"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # All transactions amount distribution
        ax1.hist(df['Amount'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Transaction Amount')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Transaction Amount Distribution (All)', fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Fraud transactions amount distribution
        fraud_amounts = df[df['Class'] == 1]['Amount']
        ax2.hist(fraud_amounts, bins=30, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Transaction Amount')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Transaction Amount Distribution (Fraud)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [df[df['Class'] == 0]['Amount'], df[df['Class'] == 1]['Amount']]
        ax3.boxplot(data_to_plot, labels=['Legitimate', 'Fraud'])
        ax3.set_ylabel('Transaction Amount')
        ax3.set_title('Transaction Amount - Box Plot', fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "amount_distribution.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print amount statistics
        print("\nAmount Statistics:")
        print(f"All transactions - Mean: ${df['Amount'].mean():.2f}, Max: ${df['Amount'].max():.2f}")
        print(f"Fraud transactions - Mean: ${fraud_amounts.mean():.2f}, Max: ${fraud_amounts.max():.2f}")
    
    def plot_feature_distributions(self, df, num_features=12):
        """Plot distributions of first few PCA features"""
        # Select first n features
        features = [f'V{i}' for i in range(1, num_features + 1)]
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(features):
            if i < len(axes):
                # Plot distribution for legitimate transactions
                legit_data = df[df['Class'] == 0][feature]
                fraud_data = df[df['Class'] == 1][feature]
                
                axes[i].hist(legit_data, bins=50, alpha=0.6, label='Legitimate', color='blue')
                axes[i].hist(fraud_data, bins=50, alpha=0.6, label='Fraud', color='red')
                axes[i].set_title(f'Distribution of {feature}', fontweight='bold')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(features), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, df):
        """Plot correlation heatmap"""
        # Calculate correlation matrix
        corr_matrix = df.corr()
        
        # Create figure
        plt.figure(figsize=(16, 12))
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top correlations with Class
        class_correlations = corr_matrix['Class'].sort_values(ascending=False)
        print("\nTop 10 features correlated with Class:")
        for feature, corr in class_correlations.head(11).items():
            if feature != 'Class':
                print(f"{feature}: {corr:.4f}")

# Usage
if __name__ == "__main__":
    visualizer = DataExplorationVisualizer("data")
    df = visualizer.load_data()
    
    # Generate all exploratory plots
    visualizer.plot_class_distribution(df)
    visualizer.plot_time_distribution(df)
    visualizer.plot_amount_distribution(df)
    visualizer.plot_feature_distributions(df)
    visualizer.plot_correlation_heatmap(df)
    