import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class WineQualityPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and explore the dataset"""
        self.df = pd.read_csv(self.data_path)
        print("Dataset Shape:", self.df.shape)
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst 5 rows:")
        print(self.df.head())
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        # Quality distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        self.df['quality'].hist(bins=10, edgecolor='black')
        plt.title('Wine Quality Distribution')
        plt.xlabel('Quality Score')
        plt.ylabel('Frequency')
        
        # Correlation heatmap
        plt.subplot(1, 2, 2)
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        # Quality vs alcohol
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='quality', y='alcohol', data=self.df)
        plt.title('Alcohol Content vs Wine Quality')
        plt.show()
        
        return self.df
    
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        # Separate features and target
        self.X = self.df.drop(['quality', 'Id'], axis=1)
        self.y = self.df['quality']
        
        # Handle quality as binary classification (good/bad) for better performance
        # You can modify this to keep it as multi-class
        self.y = (self.y > 5).astype(int)  # 1 for good quality (6-10), 0 for bad quality (0-5)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")
        print(f"Good wine proportion: {self.y.mean():.2f}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple classifier models"""
        print("\n=== MODEL TRAINING ===")
        
        # Define models
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SGD Classifier': SGDClassifier(random_state=42),
            'SVC': SVC(random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            if name == 'SGD Classifier':
                # SGD and SVC work better with scaled data
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.decision_function(self.X_test_scaled)
            elif name == 'SVC':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.decision_function(self.X_test_scaled)
            else:
                # Random Forest doesn't necessarily need scaling
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate accuracy
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Cross-validation score
            if name in ['SGD Classifier', 'SVC']:
                cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5)
            else:
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print(f"{name} CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n=== MODEL EVALUATION ===")
        
        # Create comparison table
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[model]['accuracy'] for model in self.results],
            'CV Mean': [self.results[model]['cv_mean'] for model in self.results],
            'CV Std': [self.results[model]['cv_std'] for model in self.results]
        })
        
        print("\nModel Comparison:")
        print(results_df.sort_values('Accuracy', ascending=False))
        
        # Plot accuracy comparison
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        models = list(self.results.keys())
        accuracies = [self.results[model]['accuracy'] for model in models]
        plt.bar(models, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Plot confusion matrices
        plt.subplot(1, 2, 2)
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_predictions = self.results[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.show()
        
        # Detailed classification report for best model
        print(f"\nDetailed Report for Best Model ({best_model_name}):")
        print(classification_report(self.y_test, best_predictions, 
                                target_names=['Bad Quality', 'Good Quality']))
    
    def feature_importance(self):
        """Analyze feature importance (for Random Forest)"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            features = self.X.columns
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance - Random Forest')
            plt.tight_layout()
            plt.show()
            
            print("\nFeature Importance Ranking:")
            print(importance_df)
    
    def predict_new_wine(self, wine_features):
        """Predict quality for new wine samples"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        best_model = self.results[best_model_name]['model']
        
        # Scale features if using SGD or SVC
        if best_model_name in ['SGD Classifier', 'SVC']:
            wine_features_scaled = self.scaler.transform([wine_features])
            prediction = best_model.predict(wine_features_scaled)[0]
            probability = best_model.decision_function(wine_features_scaled)[0]
        else:
            prediction = best_model.predict([wine_features])[0]
            probability = best_model.predict_proba([wine_features])[0, 1]
        
        quality = "Good" if prediction == 1 else "Bad"
        confidence = abs(probability)
        
        print(f"\nPrediction: {quality} Quality")
        print(f"Confidence Score: {confidence:.4f}")
        
        return prediction, probability

def main():
    """Main function to run the wine quality prediction pipeline"""
    # Initialize the predictor
    predictor = WineQualityPredictor('data/WineQT.csv')
    
    # Execute the pipeline
    predictor.load_data()
    predictor.explore_data()
    predictor.preprocess_data()
    predictor.train_models()
    predictor.evaluate_models()
    predictor.feature_importance()
    
    # Example prediction
    print("\n=== EXAMPLE PREDICTION ===")
    # Example wine features (you can modify these values)
    example_wine = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
    predictor.predict_new_wine(example_wine)

if __name__ == "__main__":
    main()
    