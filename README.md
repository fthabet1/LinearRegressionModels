---
title: Linear Regression Library
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.32.2"
app_file: app.py
pinned: false
---

# 🧠 Linear Regression Models from Scratch — Visualized

An interactive Streamlit app that showcases univariate, multivariate, Lasso, and Ridge regression models built entirely from scratch in Python. Visualize model performance, explore cleaned Kaggle datasets, and understand the impact of regularization — all in your browser.

# Linear Regression Models

This project demonstrates various linear regression models with a Streamlit web interface.

## Models Implemented

1. **Univariate Linear Regression**: Simple linear regression with one feature
2. **Multivariate Linear Regression**: Linear regression with multiple features
3. **Ridge Regression**: Linear regression with L2 regularization
4. **Lasso Regression**: Linear regression with L1 regularization

## Running the Application

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Deployment Notes

For deployment on Hugging Face Spaces, the app.py file has been modified to include all necessary model code directly in the file, avoiding any import issues that may occur in the Hugging Face Spaces environment. This monolithic approach ensures the application runs smoothly regardless of Python path configurations.

The `app.py` file is completely self-contained and does not rely on imports from the LinearRegression package, making it robust for deployment in constrained environments.

## Project Structure

- `app.py`: Self-contained application with all model classes defined internally
- `LinearRegression/`: Original package containing model implementations (used for local development/extension)
  - `models/`: Linear regression model implementations
  - `preprocessing/`: Data preprocessing utilities
  - `optimizers/`: Optimization algorithms
  - `utils/`: Utility functions

## Datasets

Example datasets are included in the `datasets/` directory.