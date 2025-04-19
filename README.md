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

### On Hugging Face Spaces
If you're deploying this on Hugging Face Spaces, use the fixed version of the app:

```bash
streamlit run app_fixed.py
```

The `app_fixed.py` file contains special import handling to work around import issues on Hugging Face Spaces.

## Project Structure

- `app.py`: Main application file
- `app_fixed.py`: Version optimized for Hugging Face Spaces
- `LinearRegression/`: Package containing model implementations
  - `models/`: Linear regression model implementations
  - `preprocessing/`: Data preprocessing utilities
  - `optimizers/`: Optimization algorithms
  - `utils/`: Utility functions

## Datasets

Example datasets are included in the `datasets/` directory.