# create_project_structure.py
import os
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    base_dir = Path("credit-card-fraud-detection")
    
    # Main directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "src/data_preprocessing",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/monitoring",
        "src/utils",
        "notebooks",
        "tests",
        "models",
        "results/figures/confusion_matrices",
        "results/figures/roc_curves", 
        "results/figures/feature_importance",
        "results/reports",
        "results/predictions",
        "docs",
        "configuration",
        ".github/workflows"
    ]
    
    # Create all directories
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {base_dir / directory}")
    
    # Create __init__.py files
    init_dirs = [
        "src", "src/data_preprocessing", "src/models", "src/training",
        "src/evaluation", "src/monitoring", "src/utils", "tests"
    ]
    
    for init_dir in init_dirs:
        (base_dir / init_dir / "__init__.py").touch()
        print(f"Created: {base_dir / init_dir / '__init__.py'}")
    
    print("Project structure created successfully!")
    return base_dir

if __name__ == "__main__":
    create_project_structure()
    