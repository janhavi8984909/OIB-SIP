# notebooks/03_feature_engineering.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FeatureImportanceVisualizer:
    def __init__(self, results_dir="results/figures/feature_importance"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_feature_importance_comparison(self, feature_importances):
        """Plot feature importance comparison across different models"""
        n_models = len(feature_importances)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance_df) in enumerate(feature_importances.items()):
            # Plot top 15 features
            top_features = importance_df.head(15)
            
            axes[i].barh(range(len(top_features)), top_features['importance'], 
                        color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
            axes[i].set_yticks(range(len(top_features)))
            axes[i].set_yticklabels(top_features['feature'])
            axes[i].set_xlabel('Importance Score')
            axes[i].set_title(f'Feature Importance - {model_name}', fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "feature_importance_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_combined_feature_importance(self, feature_importances, top_n=15):
        """Plot combined feature importance across all models"""
        # Combine importances from all models
        combined_importance = {}
        
        for model_name, importance_df in feature_importances.items():
            for _, row in importance_df.head(top_n).iterrows():
                feature = row['feature']
                importance = row['importance']
                
                if feature in combined_importance:
                    combined_importance[feature] += importance
                else:
                    combined_importance[feature] = importance
        
        # Normalize and sort
        combined_df = pd.DataFrame({
            'feature': list(combined_importance.keys()),
            'combined_importance': list(combined_importance.values())
        })
        combined_df = combined_df.sort_values('combined_importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(combined_df['feature'].tail(top_n), 
                combined_df['combined_importance'].tail(top_n),
                color=plt.cm.plasma(np.linspace(0, 1, top_n)))
        
        plt.xlabel('Combined Importance Score')
        plt.title(f'Top {top_n} Features - Combined Importance Across Models', 
                 fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plt.savefig(self.results_dir / "combined_feature_importance.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return combined_df

# Example usage with sample data
if __name__ == "__main__":
    # Sample feature importance data
    feature_importances = {
        'Random Forest': pd.DataFrame({
            'feature': [f'V{i}' for i in range(1, 21)] + ['Amount', 'Time'],
            'importance': np.random.exponential(1, 22)
        }).sort_values('importance', ascending=False),
        
        'Logistic Regression': pd.DataFrame({
            'feature': [f'V{i}' for i in range(1, 21)] + ['Amount', 'Time'],
            'importance': np.random.exponential(1, 22)
        }).sort_values('importance', ascending=False)
    }
    
    visualizer = FeatureImportanceVisualizer()
    visualizer.plot_feature_importance_comparison(feature_importances)
    combined_df = visualizer.plot_combined_feature_importance(feature_importances)
    