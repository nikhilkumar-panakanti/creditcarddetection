"""
Simple Flask Web Application for Credit Card Default Prediction
Run this with: python web_app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from model_predictor import CreditDefaultPredictor
import pandas as pd

app = Flask(__name__)

# Initialize predictor once when app starts
predictor = CreditDefaultPredictor()

@app.route('/')
def home():
    """Main page with input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        # Get form data
        data = request.get_json()
        
        # Calculate engineered features
        input_data = {
            # Basic Information
            'LIMIT_BAL': float(data['LIMIT_BAL']),
            'SEX': int(data['SEX']),
            'EDUCATION': int(data['EDUCATION']),
            'MARRIAGE': int(data['MARRIAGE']),
            'AGE': int(data['AGE']),
            
            # Payment Status
            'PAY_0': int(data['PAY_0']),
            'PAY_2': int(data['PAY_2']),
            'PAY_3': int(data['PAY_3']),
            'PAY_4': int(data['PAY_4']),
            'PAY_5': int(data['PAY_5']),
            'PAY_6': int(data['PAY_6']),
            
            # Bill Amounts
            'BILL_AMT1': float(data['BILL_AMT1']),
            'BILL_AMT2': float(data['BILL_AMT2']),
            'BILL_AMT3': float(data['BILL_AMT3']),
            'BILL_AMT4': float(data['BILL_AMT4']),
            'BILL_AMT5': float(data['BILL_AMT5']),
            'BILL_AMT6': float(data['BILL_AMT6']),
            
            # Payment Amounts
            'PAY_AMT1': float(data['PAY_AMT1']),
            'PAY_AMT2': float(data['PAY_AMT2']),
            'PAY_AMT3': float(data['PAY_AMT3']),
            'PAY_AMT4': float(data['PAY_AMT4']),
            'PAY_AMT5': float(data['PAY_AMT5']),
            'PAY_AMT6': float(data['PAY_AMT6']),
        }
        
        # Calculate engineered features
        input_data['PAY_RATIO_1'] = input_data['PAY_AMT1'] / input_data['BILL_AMT1'] if input_data['BILL_AMT1'] > 0 else 0
        input_data['PAY_RATIO_2'] = input_data['PAY_AMT2'] / input_data['BILL_AMT2'] if input_data['BILL_AMT2'] > 0 else 0
        input_data['PAY_RATIO_3'] = input_data['PAY_AMT3'] / input_data['BILL_AMT3'] if input_data['BILL_AMT3'] > 0 else 0
        input_data['PAY_RATIO_4'] = input_data['PAY_AMT4'] / input_data['BILL_AMT4'] if input_data['BILL_AMT4'] > 0 else 0
        input_data['PAY_RATIO_5'] = input_data['PAY_AMT5'] / input_data['BILL_AMT5'] if input_data['BILL_AMT5'] > 0 else 0
        input_data['PAY_RATIO_6'] = input_data['PAY_AMT6'] / input_data['BILL_AMT6'] if input_data['BILL_AMT6'] > 0 else 0
        input_data['CREDIT_UTIL_1'] = input_data['BILL_AMT1'] / input_data['LIMIT_BAL'] if input_data['LIMIT_BAL'] > 0 else 0
        input_data['AVG_PAY_STATUS'] = (input_data['PAY_0'] + input_data['PAY_2'] + input_data['PAY_3'] + 
                                        input_data['PAY_4'] + input_data['PAY_5'] + input_data['PAY_6']) / 6
        
        # Get model choice (default to best model)
        model_name = data.get('model', predictor.get_best_model_name())
        
        # Make prediction
        result = predictor.predict(input_data, model_name=model_name)
        
        # Return results
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/models', methods=['GET'])
def get_models():
    """Return list of available models with their performance metrics."""
    models_info = predictor.get_model_info()
    best_model = predictor.get_best_model_name()
    
    return jsonify({
        'models': models_info,
        'best_model': best_model
    })

if __name__ == '__main__':
    print("="*70)
    print("Starting Credit Card Default Prediction Web Application")
    print("="*70)
    print(f"Best Model: {predictor.get_best_model_name()}")
    print("Open your browser and go to: http://localhost:5000")
    print("="*70)
    app.run(debug=True, host='0.0.0.0', port=5000)
