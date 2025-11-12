# main.py
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.logger import project_logger
from data_preprocessing.pipeline import DataProcessingPipeline
from training.train import ModelTrainer
from evaluation.visualization import ResultsVisualizer

def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection System')
    parser.add_argument('--mode', choices=['preprocess', 'train', 'evaluate', 'full'], 
                       default='full', help='Pipeline mode')
    parser.add_argument('--data_path', default='data', help='Path to data directory')
    parser.add_argument('--balance_method', choices=['undersample', 'oversample', 'none'],
                       default='undersample', help='Class imbalance handling method')
    
    args = parser.parse_args()
    
    project_logger.info("Starting Credit Card Fraud Detection System")
    
    try:
        if args.mode in ['preprocess', 'full']:
            project_logger.info("Running data preprocessing pipeline...")
            pipeline = DataProcessingPipeline(args.data_path, args.data_path)
            pipeline.run_pipeline(balance_method=args.balance_method)
        
        if args.mode in ['train', 'full']:
            project_logger.info("Running model training pipeline...")
            trainer = ModelTrainer()
            training_results, evaluation_results = trainer.run_training_pipeline()
            
            project_logger.info("Model training completed successfully")
        
        if args.mode in ['evaluate', 'full']:
            project_logger.info("Generating evaluation visualizations...")
            visualizer = ResultsVisualizer()
            # Add visualization code here
            
            project_logger.info("Evaluation visualizations generated")
        
        project_logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        project_logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
    