import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from LinearRegression.models import UnivariateLinearModel
from LinearRegression.models import MultivariateLinearModel
from LinearRegression.models import RidgeRegression
from LinearRegression.models import LassoRegression
from LinearRegression.preprocessing import FeatureNormalizer
from LinearRegression.preprocessing import trainTestSplitData

st.set_page_config(
    page_title="Linear Regression Models",
    layout="wide"
)

if 'models' not in st.session_state:
    st.session_state.models = {
        'univariate': None,
        'multivariate': None,
        'ridge': None,
        'lasso': None
    }
    
if 'normalizers' not in st.session_state:
    st.session_state.normalizers = {
        'univariate': None,
        'multivariate': None,
        'ridge': None,
        'lasso': None
    }

if 'data' not in st.session_state:
    st.session_state.data = {
        'X': None,
        'y': None,
        'X_train': None,
        'y_train': None,
        'X_test': None,
        'y_test': None,
        'feature_names': None,
        'target_name': None
    }

def resetSessionState():
    if st.button("Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def getAvailableDatasets():
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    return [f for f in os.listdir(datasets_dir) if f.endswith('.csv')]

def getFilteredDatasets(model_type):
    """Filter datasets based on the selected model type"""
    all_datasets = getAvailableDatasets()
    
    if model_type == "univariate":
        return [dataset for dataset in all_datasets if "univariate" in dataset.lower()]
    else:
        return [dataset for dataset in all_datasets if "multivariate" in dataset.lower()]


st.title("Linear Regression Models")

resetSessionState()

with st.sidebar:
    st.header("Model Configuration")
    
    modelType = st.selectbox(
        "Model Type",
        options=["univariate", "multivariate", "ridge", "lasso"],
        format_func=lambda x: {
            "univariate": "Univariate Linear Regression",
            "multivariate": "Multivariate Linear Regression",
            "ridge": "Ridge Regression",
            "lasso": "Lasso Regression"
        }[x]
    )
    
    learningRate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
    maxIterations = st.number_input("Max Iterations", min_value=10, max_value=10000, value=1000, step=100)
    
    if modelType in ["ridge", "lasso"]:
        lambdaParam = st.number_input("Lambda (Regularization)", min_value=0.0, max_value=100.0, value=1.0, step=0.1, format="%.2f")
    else:
        lambdaParam = 0.0
    
    st.header("Dataset Selection")
    filteredDatasets = getFilteredDatasets(modelType)
    selectedDataset = st.selectbox(
        "Choose a Dataset", 
        options=[""] + filteredDatasets,
        key=f"dataset_select_{modelType}"
    )

    testSize = st.slider("Test Set Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)

    # Action buttons
    if selectedDataset:
        if st.button("Load Dataset"):
            datasetPath = os.path.join('datasets', selectedDataset)
            data = pd.read_csv(datasetPath)
            
            st.session_state.loaded_df = data
            
            feature_cols = data.columns[:-1].tolist()
            target_col = data.columns[-1]
            
            st.session_state.data['feature_names'] = feature_cols
            st.session_state.data['target_name'] = target_col
            
            X = data[feature_cols].values
            y = data[target_col].values
            
            st.session_state.data['X'] = X
            st.session_state.data['y'] = y
            
            X_train, X_test, y_train, y_test = trainTestSplitData(data, target_col, testSize=testSize)
            
            st.session_state.data['X_train'] = X_train.values
            st.session_state.data['y_train'] = y_train.values
            st.session_state.data['X_test'] = X_test.values
            st.session_state.data['y_test'] = y_test.values
            
            st.session_state.models = {
                'univariate': None,
                'multivariate': None,
                'ridge': None,
                'lasso': None
            }
            
            st.session_state.normalizers = {
                'univariate': None,
                'multivariate': None,
                'ridge': None,
                'lasso': None
            }
            
            if 'costHistory' in st.session_state:
                del st.session_state['costHistory']
            
            st.success(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
    
    if st.session_state.data['X_train'] is not None:
        if st.button("Train Model"):
            try:
                X_train = st.session_state.data['X_train']
                y_train = st.session_state.data['y_train']
                
                normalizer = FeatureNormalizer()
                normalizer.fit(X_train)
                X_trainNormalized = normalizer.transform(X_train)
                
                st.session_state.normalizers[modelType] = normalizer
                
                if modelType == 'univariate':
                    model = UnivariateLinearModel(
                        learningRate=learningRate,
                        maxIterations=maxIterations
                    )
                elif modelType == 'multivariate':
                    model = MultivariateLinearModel(
                        learningRate=learningRate,
                        maxIterations=maxIterations
                    )
                elif modelType == 'ridge':
                    model = RidgeRegression(
                        learningRate=learningRate,
                        maxIterations=maxIterations,
                        lambda_=lambdaParam
                    )
                elif modelType == 'lasso':
                    model = LassoRegression(
                        learningRate=learningRate,
                        maxIterations=maxIterations,
                        lambda_=lambdaParam
                    )
                
                # Train the model
                startTime = time.time()
                model.fit(X_trainNormalized, y_train)
                while not model.isFitted:  
                    time.sleep(0.1)  
                trainingTime = time.time() - startTime
                
                # Store the trained model
                st.session_state.models[modelType] = model
                st.session_state.training_time = trainingTime
                
                st.success(f"Model trained in {trainingTime:.2f} seconds")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")

# Main content area
if 'loaded_df' in st.session_state:

    st.header("Dataset Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Dataset shape:", st.session_state.loaded_df.shape)
        st.write("Features:", st.session_state.data['feature_names'])
        st.write("Target:", st.session_state.data['target_name'])
    
    with col2:
        st.write("Sample data:")
        st.dataframe(st.session_state.loaded_df.head())

    if st.session_state.models[modelType] is not None:
        try:
            st.header("Model Performance")

            normalizer = st.session_state.normalizers[modelType]
            model = st.session_state.models[modelType]
            
            X_train = st.session_state.data['X_train']
            y_train = st.session_state.data['y_train']
            X_test = st.session_state.data['X_test']
            y_test = st.session_state.data['y_test']
            
            X_trainNormalized = normalizer.transform(X_train)
            y_trainPred = model.predict(X_trainNormalized)
            
            X_testNormalized = normalizer.transform(X_test)
            y_testPred = model.predict(X_testNormalized)
            
            trainScore = model.score(X_trainNormalized, y_train)
            testScore = model.score(X_testNormalized, y_test)
                        
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Metrics")
                st.write(f"R² Score: {trainScore:.4f}")
                st.write(f"Training Time: {st.session_state.training_time:.2f} seconds")
                
                if 'costHistory' in st.session_state:
                    costHistory = st.session_state.costHistory
                    if len(costHistory) > 0:
                        fig, ax = plt.subplots()
                        ax.plot(costHistory)
                        ax.set_title('Cost History')
                        ax.set_xlabel('Iteration')
                        ax.set_ylabel('Cost')
                        st.pyplot(fig)
                    
            with col2:
                st.subheader("Test Metrics")
                st.write(f"R² Score: {testScore:.4f}")

            
            st.header("Visualization")
            
            if modelType == 'univariate':
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(X_train, y_train, alpha=0.5, label='Training Data')
                ax.scatter(X_test, y_test, alpha=0.5, label='Test Data')
                
                xMin, x_max = np.min(st.session_state.data['X']), np.max(st.session_state.data['X'])
                xLine = np.linspace(xMin, x_max, 100).reshape(-1, 1)
                xLineNormalized = normalizer.transform(xLine)
                yLine = model.predict(xLineNormalized)
                
                ax.plot(xLine, yLine, 'r-', label='Regression Line')
                
                ax.set_xlabel(st.session_state.data['feature_names'][0])
                ax.set_ylabel(st.session_state.data['target_name'])
                ax.legend()
                ax.set_title('Linear Regression Model')
                
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                ax.scatter(y_trainPred, y_train, alpha=0.5, label='Training Data')
                ax.scatter(y_testPred, y_test, alpha=0.5, label='Test Data')
                
                yMin = min(np.min(y_train), np.min(y_test))
                yMax = max(np.max(y_train), np.max(y_test))
                ax.plot([yMin, yMax], [yMin, yMax], 'r--', label='Perfect Prediction')
                
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.legend()
                ax.set_title('Predicted vs Actual Values')
                
                st.pyplot(fig)
            
            # Show feature importance for multivariate models
            if modelType in ['multivariate', 'ridge', 'lasso']:
                st.header("Feature Importance")
                
                featureImportance = abs(model.weights.flatten())
                features = st.session_state.data['feature_names']
                
                importanceDF = pd.DataFrame({
                    'Feature': features,
                    'Importance': featureImportance
                })
                importanceDF = importanceDF.sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importanceDF, ax=ax)
                ax.set_title('Feature Importance')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in visualization: {str(e)}")
else:
    st.info("Please select a dataset from the sidebar and click 'Load Dataset' to begin.")
    
    st.header("About Linear Regression Models")
    st.write("""
    This application demonstrates various linear regression models:
    
    1. **Univariate Linear Regression**: Simple linear regression with one feature
    2. **Multivariate Linear Regression**: Linear regression with multiple features
    3. **Ridge Regression**: Linear regression with L2 regularization
    4. **Lasso Regression**: Linear regression with L1 regularization
    
    Choose a dataset and model type from the sidebar to get started.
    """)
