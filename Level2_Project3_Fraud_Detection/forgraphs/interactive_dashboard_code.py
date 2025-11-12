# notebooks/05_model_evaluation.ipynb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class InteractiveDashboard:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def create_interactive_model_comparison(self, results_dict):
        """Create interactive model comparison dashboard"""
        metrics = ['precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
        metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC AUC', 'Average Precision']
        
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metric_names + ['Performance Summary'],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "bar"}, {"type": "table"}]]
        )
        
        models = list(results_dict.keys())
        
        for i, metric in enumerate(metrics):
            row = i // 3 + 1
            col = i % 3 + 1
            
            values = [results_dict[model][metric] for model in models]
            
            fig.add_trace(
                go.Bar(name=metric_names[i], x=models, y=values,
                      marker_color=self.colors[:len(models)]),
                row=row, col=col
            )
        
        # Add summary table
        summary_data = []
        for model in models:
            row = [model]
            for metric in metrics:
                row.append(f"{results_dict[model][metric]:.3f}")
            summary_data.append(row)
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Model'] + metric_names,
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[[row[i] for row in summary_data] for i in range(len(summary_data[0]))],
                          fill_color='lavender',
                          align='left')),
            row=2, col=3
        )
        
        fig.update_layout(height=800, title_text="Interactive Model Comparison Dashboard")
        fig.show()
        
        return fig

# Usage
if __name__ == "__main__":
    # Sample data
    sample_results = {
        'Logistic Regression': {
            'precision': 0.85, 'recall': 0.78, 'f1_score': 0.81,
            'roc_auc': 0.92, 'average_precision': 0.83
        },
        'Random Forest': {
            'precision': 0.88, 'recall': 0.82, 'f1_score': 0.85,
            'roc_auc': 0.95, 'average_precision': 0.87
        },
        'Neural Network': {
            'precision': 0.87, 'recall': 0.85, 'f1_score': 0.86,
            'roc_auc': 0.94, 'average_precision': 0.86
        }
    }
    
    dashboard = InteractiveDashboard()
    fig = dashboard.create_interactive_model_comparison(sample_results)
    
    # Save interactive plot
    fig.write_html("results/figures/interactive_dashboard.html")
    