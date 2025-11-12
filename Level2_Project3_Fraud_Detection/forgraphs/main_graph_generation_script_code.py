# scripts/generate_all_graphs.py
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.logger import project_logger
from evaluation.visualization import ModelEvaluationVisualizer

def generate_all_graphs():
    """Generate all graphs for the project"""
    project_logger.info("Starting graph generation...")
    
    try:
        # Initialize visualizers
        model_viz = ModelEvaluationVisualizer()
        
        # Load evaluation results
        results = model_viz.load_evaluation_results()
        
        # Generate model evaluation graphs
        project_logger.info("Generating model evaluation graphs...")
        model_viz.plot_model_comparison(results)
        model_viz.plot_roc_curves(results)
        model_viz.plot_precision_recall_curves(results)
        model_viz.plot_confusion_matrices(results)
        
        project_logger.info("All graphs generated successfully!")
        
    except Exception as e:
        project_logger.error(f"Graph generation failed: {e}")
        raise

if __name__ == "__main__":
    generate_all_graphs()
    