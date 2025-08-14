# Enterprise Tabular Machine Learning Pipeline

A comprehensive implementation of machine learning and deep learning techniques for tabular data, covering the entire pipeline from data exploration to production deployment. This project demonstrates advanced ML practices, ensemble methods, and cloud deployment strategies.

## 🚀 Project Overview

This repository contains a complete implementation of  tabular ML techniques, comparing gradient boosting methods with deep learning approaches. The project showcases real-world applications using Airbnb pricing data and real estate datasets, demonstrating both classical and state-of-the-art methodologies.

### Key Features
- **Comprehensive ML Pipeline**: End-to-end automated pipeline with data preprocessing, model training, and deployment
- **Algorithm Comparison**: Side-by-side implementation of gradient boosting (XGBoost, LightGBM) vs. deep learning approaches
- **Production Ready**: Flask web deployment with Google Cloud Vertex AI integration
- **Explainable AI**: SHAP implementation for model interpretability
- **Advanced Techniques**: Feature engineering, hyperparameter optimization, ensemble methods

## 📊 Datasets

### Primary Datasets
- **Airbnb NYC (2019)**: Price prediction for New York City listings
- **Airbnb Tokyo**: Calendar data and pricing analysis
- **Kuala Lumpur Real Estate**: Property valuation with advanced preprocessing

### Dataset Characteristics
- Mixed data types (numerical, categorical, text)
- Missing value handling strategies
- Feature engineering opportunities
- Real-world data quality challenges

## 🏗️ Project Structure

```
├── 1. Tabular Dataset.ipynb                    # Introduction to tabular data concepts
├── 2. Machine learning vs Deep Learning/       # Comparative analysis framework
│   ├── keras/                                  # Deep learning implementation
│   ├── xgboost/                               # Gradient boosting implementation
│   └── custom_classes.py                     # Reusable utilities
├── 3. Classical Algorithms for Tabular Data.ipynb  # Scikit-learn implementations
├── 4. Decision Trees and Gradient Boosting.ipynb   # Tree-based methods
├── 5. Advanced Feature Processing Methods.ipynb    # Feature engineering techniques
├── 6. An end-to-end Example using XGBoost.ipynb   # Complete XGBoost pipeline
├── 7. deep learning with tabular data/        # Multiple DL frameworks
│   ├── fastai_basics/                         # FastAI implementation
│   ├── lightning_flash_basics/                # PyTorch Lightning Flash
│   └── tabnet_basics/                         # TabNet architecture
├── 8. Deep learning best practices/           # Advanced DL techniques
├── 9. Model Deployment/                       # Flask + Cloud deployment
├── 10. Building a machine learning pipeline/  # Automated ML pipeline
├── 11. Blending Gradient Boosting and Deep Learning.ipynb  # Ensemble methods
├── 12. K-nearest Neighbors and Support Vector Machines.ipynb  # Classical ML
└── tabular_datasets/                          # Data storage
```

## 🛠️ Technologies & Frameworks

### Machine Learning Stack
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Classical ML**: Scikit-learn (Random Forest, SVM, KNN)
- **Deep Learning**: Keras, PyTorch, fastai, TabNet, Lightning Flash
- **Feature Engineering**: Advanced preprocessing pipelines
- **Hyperparameter Optimization**: Bayesian optimization, random search

### Development & Deployment
- **Web Framework**: Flask with custom templates
- **Cloud Platform**: Google Cloud Vertex AI
- **Containerization**: Docker for model deployment
- **Explainability**: SHAP for model interpretability
- **Data Processing**: Pandas, NumPy, advanced preprocessing

### Visualization & Analysis
- **EDA Tools**: Matplotlib, Seaborn, Plotly
- **Model Analysis**: Learning curves, feature importance
- **Performance Metrics**: Custom evaluation frameworks

### Key Findings
- **Gradient Boosting**: Superior performance on structured data with faster training
- **Deep Learning**: Better handling of complex feature interactions
- **Ensemble Methods**: Best overall performance combining both approaches
- **Deployment**: Successful cloud deployment with <100ms inference time

## 🔬 Advanced Features

### Feature Engineering
- **Target Encoding**: Advanced categorical feature handling
- **Missing Data Imputation**: Multivariate imputation strategies
- **Feature Selection**: Boruta, stability selection, forward/backward selection
- **Numerical Transformations**: Box-Cox, Yeo-Johnson transformations

### Model Optimization
- **Hyperparameter Tuning**: Bayesian optimization with Optuna
- **Cross-Validation**: Stratified k-fold with time-based splits
- **Early Stopping**: Adaptive stopping criteria
- **Regularization**: L1/L2, dropout, batch normalization

### Explainability
- **SHAP Values**: Local and global model explanations
- **Feature Importance**: Tree-based and permutation importance
- **Partial Dependence**: Understanding feature relationships
- **Model Debugging**: Comprehensive analysis tools

## 🚢 Deployment Architecture

### Local Deployment
- Flask web application with custom templates
- Model serving with preprocessing pipelines
- Interactive prediction interface

### Cloud Deployment (Google Cloud Vertex AI)
- Containerized model deployment
- Auto-scaling endpoints
- Model monitoring and versioning
- CI/CD pipeline integration


## 📚 Learning Journey

This project follows a structured learning path:

1. **Foundation**: Understanding tabular data characteristics
2. **Classical ML**: Implementing traditional algorithms
3. **Advanced Methods**: Gradient boosting and ensemble techniques
4. **Deep Learning**: Modern neural network approaches
5. **Production**: End-to-end deployment and monitoring
6. **Optimization**: Advanced techniques and best practices
---
