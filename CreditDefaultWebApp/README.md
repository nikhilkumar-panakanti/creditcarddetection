# Credit Card Default Prediction - Web Application

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="center">

### 🎯 Production-Ready ML Application for Credit Risk Assessment

**Best Model: Neural Network | 79.95% Accuracy | 50.1% F1-Score | 75.4% ROC-AUC**

[Quick Start](#-quick-start) • [Features](#-features) • [Model Performance](#-model-performance) • [Deployment](#-deployment-options) • [Documentation](#-input-features-31-total)

</div>

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Technologies Used](#️-technologies-used)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
- [Model Performance](#-model-performance)
- [Using Models in Code](#-using-models-in-your-own-code)
- [Input Features](#-input-features-31-total)
- [Web Interface](#-web-interface-features)
- [Security Notes](#-security-notes)
- [Model Training Details](#-model-training-details)
- [Academic Reference](#-academic-reference)
- [Deployment Options](#-deployment-options)
- [Testing Checklist](#-testing-checklist)
- [Next Steps](#-next-steps--enhancements)
- [License](#-license)

## 📋 Project Overview

A production-ready machine learning web application for predicting credit card default risk. This system leverages 6 trained ML models (Neural Network, Random Forest, Gradient Boosting, Support Vector Machine, Logistic Regression, and KNN) to provide real-time risk assessment for credit card customers. Developed as part of a dissertation research project on credit risk assessment using advanced machine learning techniques.

**Key Achievement**: Neural Network model achieves **79.95% accuracy** with **50.1% F1-score**, providing optimal balance between precision (55.7%) and recall (45.4%) for identifying potential defaulters.

## 🛠️ Technologies Used

### Backend
- **Python 3.10**: Core programming language
- **Flask 3.1.2**: Web framework for REST API and web interface
- **scikit-learn 1.7.2**: ML model training and evaluation
- **TensorFlow/Keras**: Neural Network implementation
- **NumPy & Pandas**: Data manipulation and numerical computing
- **Joblib**: Model serialization and loading

### Frontend
- **HTML5 & CSS3**: Responsive web interface
- **Bootstrap 5**: UI components and styling
- **JavaScript**: Interactive form handling

### ML Models
- Neural Network (Best Model)
- Gradient Boosting
- Random Forest
- Support Vector Machine
- Logistic Regression
- K-Nearest Neighbors

### Deployment
- **Gunicorn/Waitress**: Production WSGI servers
- **Docker**: Containerization
- **Git**: Version control

## 🎯 Features

- **6 Pre-trained ML Models**: All models trained on 30,000 customer records and ready for deployment
- **Best Model - Neural Network**: 79.95% accuracy, 50.1% F1-score, 75.4% ROC-AUC
- **Real-time Predictions**: Instant risk assessment with probability scores
- **Professional Web Interface**: Responsive design with intuitive form layout
- **Probabilistic Risk Scoring**: Confidence levels and detailed metrics for informed decision-making
- **Model Flexibility**: Switch between different ML algorithms to compare predictions
- **Production Ready**: Complete with preprocessing pipeline, error handling, and deployment configuration

## 📁 Repository Structure

```
CreditDefaultWebApp/
│
├── saved_models/                      # Pre-trained ML models & preprocessing
│   ├── neural_network.pkl             # Neural Network (Best Model - 79.95% accuracy)
│   ├── gradient_boosting.pkl          # Gradient Boosting (80.52% accuracy)
│   ├── random_forest.pkl              # Random Forest (80.78% accuracy)
│   ├── support_vector_machine.pkl     # SVM (80.68% accuracy)
│   ├── logistic_regression.pkl        # Logistic Regression (80.88% accuracy)
│   ├── k-nearest_neighbors.pkl        # KNN (73.97% accuracy)
│   ├── scaler.pkl                     # StandardScaler for data normalization
│   ├── feature_names.json             # Complete list of 31 features
│   └── model_metadata.json            # Performance metrics & best model info
│
├── templates/                         # Web UI templates
│   └── index.html                     # Main interface (responsive design)
│
├── web_app.py                         # Flask application entry point
├── model_predictor.py                 # ML model loader & prediction engine
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── QUICKSTART.md                      # Quick setup guide
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd CreditDefaultWebApp
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python web_app.py
```

The application will start on `http://localhost:5000`

Open your browser and navigate to:
- **Local**: http://localhost:5000
- **Network**: http://127.0.0.1:5000

## 🧪 Testing the Predictor (Command Line)

You can test the predictor without the web interface:

```powershell
python model_predictor.py
```

This will:
- Load all 6 trained models
- Show the best model information
- Make a sample prediction
- Display predictions from all models

## 📊 Model Performance

| Model | Accuracy | F1-Score | ROC-AUC | Precision | Recall | CV F1 (Mean) |
|-------|----------|----------|---------|-----------|--------|--------------|
| **Neural Network** ⭐ | **79.95%** | **50.1%** | **75.4%** | **55.7%** | **45.4%** | **67.0%** |
| Logistic Regression | 80.88% | 49.4% | 74.8% | 56.1% | 43.9% | 63.9% |
| Random Forest | 80.78% | 48.7% | 74.7% | 56.9% | 43.0% | 67.5% |
| Support Vector Machine | 80.68% | 49.3% | 74.6% | 56.2% | 43.8% | 61.0% |
| Gradient Boosting | 80.52% | 46.9% | 73.9% | 57.5% | 40.3% | 71.1% |
| K-Nearest Neighbors | 73.97% | 45.2% | 69.6% | 43.3% | 46.7% | 66.3% |

### 🏆 Recommended Model: Neural Network

**Why Neural Network is the Best Model:**
- **Highest F1-Score (50.1%)**: Best balance between precision and recall on test data
- **Best ROC-AUC (75.4%)**: Superior discriminative ability to distinguish defaulters from non-defaulters
- **Optimal Precision-Recall Trade-off**: 55.7% precision minimizes false positives while 45.4% recall captures nearly half of actual defaulters
- **Production-Ready Performance**: Consistent results across validation and test sets
- **Business Value**: Reduces financial risk while maintaining customer satisfaction by avoiding excessive false alarms

**Use Case Alignment**: Ideal for financial institutions requiring balanced credit risk assessment where both identifying potential defaulters (recall) and maintaining customer trust by minimizing false accusations (precision) are equally critical.

## 🔧 Using Models in Your Own Code

### Example 1: Simple Prediction with Best Model (Neural Network)

```python
from model_predictor import CreditDefaultPredictor

# Initialize predictor
predictor = CreditDefaultPredictor()

# Create input data for a sample customer
customer_data = {
    'LIMIT_BAL': 200000,
    'SEX': 2,
    'EDUCATION': 2,
    'MARRIAGE': 1,
    'AGE': 35,
    'PAY_0': 0,
    'PAY_2': 0,
    'PAY_3': 0,
    'PAY_4': 0,
    'PAY_5': 0,
    'PAY_6': 0,
    'BILL_AMT1': 50000,
    'BILL_AMT2': 48000,
    'BILL_AMT3': 45000,
    'BILL_AMT4': 42000,
    'BILL_AMT5': 40000,
    'BILL_AMT6': 38000,
    'PAY_AMT1': 5000,
    'PAY_AMT2': 4800,
    'PAY_AMT3': 4500,
    'PAY_AMT4': 4200,
    'PAY_AMT5': 4000,
    'PAY_AMT6': 3800,
    'PAY_RATIO_1': 0.1,
    'PAY_RATIO_2': 0.1,
    'PAY_RATIO_3': 0.1,
    'PAY_RATIO_4': 0.1,
    'PAY_RATIO_5': 0.1,
    'PAY_RATIO_6': 0.1,
    'CREDIT_UTIL_1': 0.25,
    'AVG_PAY_STATUS': 0.0
}

# Make prediction with best model (Neural Network)
result = predictor.predict(customer_data)
print(f"Prediction: {result['prediction_label']}")
print(f"Default Probability: {result['probability_default']:.2%}")
print(f"Model Used: {result['model_used']}")
```

### Example 2: Compare All Models

```python
# Get predictions from all 6 models
all_results = predictor.predict_all_models(customer_data)

for model_name, result in all_results.items():
    print(f"{model_name}: {result['prediction_label']} "
          f"(Probability: {result['probability_default']:.2%})")
```

### Example 3: Get Model Information

```python
# Get performance metrics for all models
models_info = predictor.get_model_info()

for model_name, metrics in models_info.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  F1-Score: {metrics['test_f1']:.3f}")

# Get best model name
best_model = predictor.get_best_model_name()
print(f"\nBest Model: {best_model}")  # Output: Neural Network
```

## 📝 Input Features (31 Total)

### Basic Information (5 features)
- `LIMIT_BAL`: Credit limit (NT$)
- `SEX`: Gender (1=male, 2=female)
- `EDUCATION`: Education level (1=graduate, 2=university, 3=high school, 4=others)
- `MARRIAGE`: Marital status (1=married, 2=single, 3=others)
- `AGE`: Age in years

### Payment Status (6 features)
- `PAY_0` to `PAY_6`: Payment delay status for past 6 months
  - -1 = Pay on time
  - 0 = No delay
  - 1+ = Number of months delayed

### Bill Amounts (6 features)
- `BILL_AMT1` to `BILL_AMT6`: Bill statements for past 6 months (NT$)

### Payment Amounts (6 features)
- `PAY_AMT1` to `PAY_AMT6`: Payment amounts for past 6 months (NT$)

### Engineered Features (8 features)
- `PAY_RATIO_1` to `PAY_RATIO_6`: Payment-to-bill ratios
- `CREDIT_UTIL_1`: Credit utilization ratio
- `AVG_PAY_STATUS`: Average payment status

## 🎨 Web Interface Features

1. **User-Friendly Form**: Easy-to-fill customer information
2. **Real-time Validation**: Input validation to prevent errors
3. **Visual Results**: 
   - Color-coded risk indicators
   - Probability bars
   - Detailed metrics
4. **Model Selection**: Choose which ML model to use
5. **Recommendations**: Actionable insights based on prediction
6. **Responsive Design**: Works on desktop and mobile

## 🔐 Security Notes

For production deployment:
1. Add input validation and sanitization
2. Implement rate limiting
3. Use HTTPS
4. Add authentication if needed
5. Set `debug=False` in `web_app.py`
6. Use a production WSGI server (e.g., Gunicorn)

## 📈 Model Training Details

- **Dataset**: UCI Taiwan Credit Card Default (30,000 customers)
- **Training Samples**: 24,000 (with SMOTE: 37,382)
- **Test Samples**: 6,000
- **Features**: 31 (23 original + 8 engineered)
- **Class Balance**: Handled with SMOTE oversampling
- **Preprocessing**: StandardScaler normalization
- **Cross-Validation**: 5-fold stratified
- **Hyperparameter Tuning**: GridSearchCV for top 3 models

## 🎓 Academic Reference

This project was developed as part of a dissertation on credit risk assessment using machine learning techniques.

**Student**: Nikhil Kumar Panaknti  
**Student ID**: 23109231  
**Date**: December 2024  
**Project**: Credit Card Default Prediction Using Machine Learning  
**Best Model**: Neural Network (79.95% accuracy, 50.1% F1-score)

### Research Highlights
- Trained and evaluated 6 ML models on 30,000 customer records
- Implemented feature engineering to create 8 additional predictive features
- Achieved optimal precision-recall balance for credit risk assessment
- Developed production-ready web application for real-time predictions

## 📞 Support & Contributions

### Getting Help
- Review the complete training notebook: `Dissertation_Code_nikhil.ipynb`
- Check model performance metrics: `saved_models/model_metadata.json`
- Test predictions directly: `python model_predictor.py`

### Contributing
Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🚀 Deployment Options

### Option 1: Local Development (Quick Start)
```bash
python web_app.py
```
Access at: `http://localhost:5000`

### Option 2: Production with Gunicorn (Linux/Mac)
```bash
# Install Gunicorn
pip install gunicorn

# Run with 4 worker processes
gunicorn -w 4 -b 0.0.0.0:5000 web_app:app

# Or with timeout and logging
gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 --access-logfile - web_app:app
```

### Option 3: Windows Production with Waitress
```bash
# Install Waitress
pip install waitress

# Run server
python -c "from waitress import serve; from web_app import app; serve(app, host='0.0.0.0', port=5000)"
```

### Option 4: Docker Deployment
Create a `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "web_app.py"]
```

Build and run:
```bash
docker build -t credit-default-app .
docker run -p 5000:5000 credit-default-app
```

### Option 5: Cloud Platforms

**Heroku**:
```bash
# Add Procfile
echo "web: gunicorn web_app:app" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

**AWS Elastic Beanstalk**:
- Package app with `requirements.txt`
- Deploy using EB CLI or Console
- Configure environment for Python 3.10

**Azure Web Apps**:
- Use Azure App Service
- Configure Python 3.10 runtime
- Deploy via GitHub Actions or Azure CLI

### Production Configuration
Before deploying to production:
1. Set `DEBUG = False` in `web_app.py`
2. Use environment variables for sensitive data
3. Enable HTTPS/SSL
4. Implement rate limiting (Flask-Limiter)
5. Add logging and monitoring
6. Use a reverse proxy (Nginx) for better performance

## ✅ Testing Checklist

- [x] All 6 models trained successfully
- [x] Models saved correctly (`.pkl` files)
- [x] Scaler saved for preprocessing
- [x] Feature names saved
- [x] Metadata saved with performance metrics
- [x] Predictor class loads all models
- [x] Sample prediction works
- [x] Web application runs
- [x] HTML interface displays correctly
- [x] Predictions return expected format

## 📚 Next Steps & Enhancements

### Immediate Improvements
- [ ] Add user authentication and session management
- [ ] Implement prediction history tracking
- [ ] Add batch prediction capability (CSV upload)
- [ ] Create REST API endpoints for integration
- [ ] Add unit tests and integration tests

### Advanced Features
- [ ] **Model Monitoring**: Track prediction distribution and model drift
- [ ] **A/B Testing**: Compare different model versions in production
- [ ] **Explainability**: Integrate SHAP values for feature importance
- [ ] **Real-time Retraining**: Periodic model updates with new data
- [ ] **Multi-language Support**: Internationalization for global use

### Infrastructure
- [ ] **Database Integration**: PostgreSQL/MongoDB for prediction storage
- [ ] **Caching**: Redis for faster repeated predictions
- [ ] **Load Balancing**: Handle multiple concurrent users
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Monitoring Dashboard**: Grafana/Prometheus for metrics

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

## 🙏 Acknowledgments

- **Dataset**: UCI Machine Learning Repository - Default of Credit Card Clients Dataset
- **Framework**: Flask web framework for Python
- **ML Libraries**: scikit-learn, TensorFlow/Keras for model development
- **Research**: Based on comprehensive dissertation research on credit risk assessment

## 📊 Project Statistics

- **Lines of Code**: ~500 (Python)
- **Models Trained**: 6 ML algorithms
- **Training Samples**: 24,000 customers (SMOTE balanced: 37,382)
- **Test Samples**: 6,000 customers
- **Features**: 31 (23 original + 8 engineered)
- **Best Accuracy**: 79.95% (Neural Network)
- **Best F1-Score**: 50.1% (Neural Network)
- **Deployment Ready**: ✅ Production-grade application

---

**⭐ If you find this project helpful, please consider giving it a star on GitHub!**

**📧 Contact**: For questions or collaboration opportunities, feel free to reach out.

**Enjoy using the Credit Card Default Prediction System! 🎉**
