import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A class for training and optimizing machine learning models for sentiment analysis
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model trainer
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
    
    def get_model_configs(self):
        """
        Get configuration for all models to train
        
        Returns:
            dict: Model configurations
        """
        return {
            'Naive_Bayes': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', MultinomialNB())
                ]),
                'params': {
                    'tfidf__max_features': [1000, 2000, 3000],
                    'tfidf__ngram_range': [(1, 1), (1, 2)],
                    'tfidf__min_df': [1, 2],
                    'tfidf__max_df': [0.7, 0.8, 0.9],
                    'clf__alpha': [0.1, 0.5, 1.0]
                }
            },
            'Logistic_Regression': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', LogisticRegression(random_state=self.random_state))
                ]),
                'params': {
                    'tfidf__max_features': [2000, 3000],
                    'tfidf__ngram_range': [(1, 2)],
                    'clf__C': [0.1, 1, 10],
                    'clf__max_iter': [1000],
                    'clf__solver': ['liblinear', 'saga']
                }
            },
            'Random_Forest': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', RandomForestClassifier(random_state=self.random_state))
                ]),
                'params': {
                    'tfidf__max_features': [2000, 3000],
                    'clf__n_estimators': [100, 200],
                    'clf__max_depth': [10, 20, None],
                    'clf__min_samples_split': [2, 5],
                    'clf__min_samples_leaf': [1, 2]
                }
            },
            'SVM': {
                'pipeline': Pipeline([
                    ('tfidf', TfidfVectorizer()),
                    ('clf', SVC(random_state=self.random_state, probability=True))
                ]),
                'params': {
                    'tfidf__max_features': [2000, 3000],
                    'clf__C': [0.1, 1, 10],
                    'clf__kernel': ['linear', 'rbf'],
                    'clf__gamma': ['scale', 'auto']
                }
            }
        }
    
    def train_models(self, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Train multiple models with hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_jobs (int): Number of parallel jobs
            
        Returns:
            dict: Trained models and their results
        """
        model_configs = self.get_model_configs()
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*60}")
            print(f"Training {model_name}...")
            print(f"{'='*60}")
            
            pipeline = config['pipeline']
            params = config['params']
            
            # Perform grid search
            grid_search = GridSearchCV(
                pipeline, 
                params, 
                cv=cv, 
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store results
            self.models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_,
                'grid_search': grid_search
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV {scoring}: {grid_search.best_score_:.4f}")
            
            # Update best model
            if grid_search.best_score_ > self.best_score:
                self.best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name
        
        print(f"\nBest model: {self.best_model_name} with {scoring}: {self.best_score:.4f}")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        
        print(f"\n{'='*60}")
        print("MODEL EVALUATION ON TEST SET")
        print(f"{'='*60}")
        
        for model_name, model_info in self.models.items():
            print(f"\n{model_name}:")
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'probabilities': y_prob,
                'classification_report': report,
                'model': model
            }
            
            print(f"Test Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(report)
        
        return results
    
    def get_model_comparison(self, test_results):
        """
        Create model comparison dataframe
        
        Args:
            test_results (dict): Test evaluation results
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name in self.models.keys():
            cv_score = self.models[model_name]['best_score']
            test_accuracy = test_results[model_name]['accuracy']
            
            comparison_data.append({
                'Model': model_name,
                'CV_Score': cv_score,
                'Test_Accuracy': test_accuracy,
                'Difference': test_accuracy - cv_score
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test_Accuracy', ascending=False)
        
        return comparison_df
    
    def save_models(self, save_dir='models/trained_models/'):
        """
        Save trained models to disk
        
        Args:
            save_dir (str): Directory to save models
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("\nSaving trained models...")
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            filename = f"{save_dir}/{model_name}_model.pkl"
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
        
        # Save best model separately
        if self.best_model:
            best_model_filename = f"{save_dir}/best_model.pkl"
            joblib.dump(self.best_model, best_model_filename)
            print(f"Best model ({self.best_model_name}) saved to {best_model_filename}")
        
        # Save training summary
        training_summary = {
            'best_model': self.best_model_name,
            'best_score': self.best_score,
            'model_details': {
                name: {
                    'best_params': info['best_params'],
                    'best_score': info['best_score']
                } for name, info in self.models.items()
            }
        }
        
        import json
        summary_filename = f"{save_dir}/training_summary.json"
        with open(summary_filename, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"Training summary saved to {summary_filename}")