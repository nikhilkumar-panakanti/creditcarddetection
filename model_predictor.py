"""
Credit Card Default Prediction - Model Loader and Predictor
This script demonstrates how to load and use the trained models for predictions.
Use this as a reference for building your web application.
"""

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path


class CreditDefaultPredictor:
    """
    A class to load trained models and make predictions on credit card default.
    """
    
    def __init__(self, models_dir='saved_models'):
        """
        Initialize the predictor by loading models and metadata.
        
        Args:
            models_dir (str): Path to directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.metadata = None
        
        self._load_models()
        
    def _load_models(self):
        """Load all trained models, scaler, and metadata."""
        print("Loading models and preprocessing components...")
        
        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        print(f"✓ Loaded scaler from {scaler_path}")
        
        # Load feature names
        features_path = self.models_dir / 'feature_names.json'
        with open(features_path, 'r') as f:
            self.feature_names = json.load(f)
        print(f"✓ Loaded {len(self.feature_names)} feature names")
        
        # Load metadata
        metadata_path = self.models_dir / 'model_metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        print(f"✓ Loaded metadata (best model: {self.metadata['best_model']})")
        
        # Load all model files
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'Gradient Boosting': 'gradient_boosting.pkl',
            'Support Vector Machine': 'support_vector_machine.pkl',
            'Neural Network': 'neural_network.pkl',
            'K-Nearest Neighbors': 'k-nearest_neighbors.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            self.models[model_name] = joblib.load(model_path)
            print(f"✓ Loaded {model_name}")
        
        print(f"\n{'='*70}")
        print(f"All models loaded successfully!")
        print(f"{'='*70}\n")
    
    def get_model_info(self, model_name=None):
        """
        Get performance information about a specific model or all models.
        
        Args:
            model_name (str, optional): Name of the model. If None, returns all.
        
        Returns:
            dict: Model performance metrics
        """
        if model_name:
            return self.metadata['models'].get(model_name)
        return self.metadata['models']
    
    def get_best_model_name(self):
        """Get the name of the best performing model."""
        return self.metadata['best_model']
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for prediction.
        
        Args:
            input_data (dict or pd.DataFrame): Raw input features
        
        Returns:
            np.ndarray: Scaled features ready for prediction
        """
        # Convert dict to DataFrame if necessary
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features correctly
        input_data = input_data[self.feature_names]
        
        # Scale the features
        scaled_data = self.scaler.transform(input_data)
        
        return scaled_data
    
    def predict(self, input_data, model_name=None, return_proba=True):
        """
        Make prediction using specified model or best model.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            model_name (str, optional): Model to use. Uses best model if None.
            return_proba (bool): If True, return probability scores
        
        Returns:
            dict: Prediction results including class and probability
        """
        # Use best model if not specified
        if model_name is None:
            model_name = self.get_best_model_name()
        
        # Get the model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        # Preprocess input
        X = self.preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        result = {
            'model_used': model_name,
            'prediction': int(prediction),
            'prediction_label': 'Default' if prediction == 1 else 'No Default'
        }
        
        # Add probability if available
        if return_proba and hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            result['probability_no_default'] = float(probabilities[0])
            result['probability_default'] = float(probabilities[1])
            result['confidence'] = float(max(probabilities))
        
        return result
    
    def predict_all_models(self, input_data, return_proba=True):
        """
        Get predictions from all models.
        
        Args:
            input_data (dict or pd.DataFrame): Input features
            return_proba (bool): If True, return probability scores
        
        Returns:
            dict: Predictions from all models
        """
        results = {}
        X = self.preprocess_input(input_data)
        
        for model_name, model in self.models.items():
            prediction = model.predict(X)[0]
            
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Default' if prediction == 1 else 'No Default'
            }
            
            if return_proba and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                result['probability_no_default'] = float(probabilities[0])
                result['probability_default'] = float(probabilities[1])
                result['confidence'] = float(max(probabilities))
            
            results[model_name] = result
        
        return results


def create_sample_input():
    """
    Create a sample input for testing.
    This shows the format needed for predictions.
    """
    sample = {
        # Basic Information
        'LIMIT_BAL': 200000,      # Credit limit
        'SEX': 2,                  # 1=male, 2=female
        'EDUCATION': 2,            # 1=graduate, 2=university, 3=high school, 4=others
        'MARRIAGE': 1,             # 1=married, 2=single, 3=others
        'AGE': 35,                 # Age in years
        
        # Payment Status (months ago: -1=pay duly, 1=delay 1 month, 2=delay 2 months, etc.)
        'PAY_0': 0,               # Payment status in September
        'PAY_2': 0,               # Payment status in August
        'PAY_3': 0,               # Payment status in July
        'PAY_4': 0,               # Payment status in June
        'PAY_5': -1,              # Payment status in May
        'PAY_6': -1,              # Payment status in April
        
        # Bill Amounts
        'BILL_AMT1': 50000,       # Bill amount in September
        'BILL_AMT2': 48000,       # Bill amount in August
        'BILL_AMT3': 45000,       # Bill amount in July
        'BILL_AMT4': 43000,       # Bill amount in June
        'BILL_AMT5': 40000,       # Bill amount in May
        'BILL_AMT6': 38000,       # Bill amount in April
        
        # Payment Amounts
        'PAY_AMT1': 2000,         # Payment in September
        'PAY_AMT2': 3000,         # Payment in August
        'PAY_AMT3': 2500,         # Payment in July
        'PAY_AMT4': 3000,         # Payment in June
        'PAY_AMT5': 2000,         # Payment in May
        'PAY_AMT6': 2000,         # Payment in April
        
        # Engineered Features (calculate these based on above)
        'PAY_RATIO_1': 2000/50000 if 50000 > 0 else 0,
        'PAY_RATIO_2': 3000/48000 if 48000 > 0 else 0,
        'PAY_RATIO_3': 2500/45000 if 45000 > 0 else 0,
        'PAY_RATIO_4': 3000/43000 if 43000 > 0 else 0,
        'PAY_RATIO_5': 2000/40000 if 40000 > 0 else 0,
        'PAY_RATIO_6': 2000/38000 if 38000 > 0 else 0,
        'CREDIT_UTIL_1': 50000/200000,
        'AVG_PAY_STATUS': (0 + 0 + 0 + 0 + (-1) + (-1)) / 6
    }
    
    return sample


def main():
    """Example usage of the predictor."""
    
    print("="*70)
    print("CREDIT CARD DEFAULT PREDICTION - DEMO")
    print("="*70)
    print()
    
    # Initialize predictor
    predictor = CreditDefaultPredictor()
    
    # Show best model info
    best_model = predictor.get_best_model_name()
    best_model_info = predictor.get_model_info(best_model)
    print(f"Best Model: {best_model}")
    print(f"  Accuracy: {best_model_info['accuracy']:.4f}")
    print(f"  F1-Score: {best_model_info['f1_score']:.4f}")
    print(f"  ROC-AUC: {best_model_info['roc_auc']:.4f}")
    print()
    
    # Create sample input
    print("="*70)
    print("SAMPLE PREDICTION")
    print("="*70)
    sample_input = create_sample_input()
    
    # Make prediction with best model
    result = predictor.predict(sample_input)
    print(f"\nUsing: {result['model_used']}")
    print(f"Prediction: {result['prediction_label']}")
    if 'probability_default' in result:
        print(f"Probability of Default: {result['probability_default']:.2%}")
        print(f"Probability of No Default: {result['probability_no_default']:.2%}")
        print(f"Confidence: {result['confidence']:.2%}")
    
    # Get predictions from all models
    print("\n" + "="*70)
    print("PREDICTIONS FROM ALL MODELS")
    print("="*70)
    all_results = predictor.predict_all_models(sample_input)
    
    for model_name, result in all_results.items():
        prob_str = f" (Default: {result['probability_default']:.2%})" if 'probability_default' in result else ""
        print(f"{model_name:25} → {result['prediction_label']}{prob_str}")
    
    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
