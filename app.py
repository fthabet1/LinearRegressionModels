import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# NOTE: This file contains a monolithic implementation of all model components
# to avoid import issues with Hugging Face Spaces. All required classes are defined
# directly within this file rather than importing from the LinearRegression package.

# ================== Model Definitions ==================
# Define BaseModel class directly in app.py
class BaseModel(ABC):
    def __init__(self):
        self.weights = None
        self.bias = None
        self.isFitted = False

    def getWeights(self):
        return self.weights
    
    def getBias(self):
        return self.bias
    
    def getParams(self):
        return self.weights, self.bias
    
    def setParams(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def score(self, X, y):
        isPandas = isinstance(y, pd.DataFrame) or isinstance(y, pd.Series)
        
        if isPandas:
            y = y.values
        else:
            y = np.array(y)

        predictions = self.predict(X)
        
        if isinstance(predictions, pd.DataFrame) or isinstance(predictions, pd.Series):
            predictions = predictions.values
        else:
            predictions = np.array(predictions)
            
        y = y.flatten()
        predictions = predictions.flatten()

        y_mean = np.mean(y)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        if ss_tot == 0:
            return 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        
        return max(0.0, r2) if not np.isnan(r2) else 0.0
    
    def validateData(self, X, y=None):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        else:
            X = np.array(X)

        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        if y is not None:
            if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
                y = y.values
            else:
                y = np.array(y)
                
            y = y.flatten()

            if X.shape[0] != y.shape[0]:
                raise ValueError(f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}.")

        return X, y if y is not None else None

# Define FeatureNormalizer class directly in app.py
class FeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std == 0] = 1
        return self
        
    def transform(self, X):
        if self.mean is None or self.std is None:
            raise Exception("Normalizer has not been fitted.")
            
        return (X - self.mean) / self.std
        
    def fitTransform(self, X):
        self.fit(X)
        return self.transform(X)

# Define GradientDescent class directly in app.py
class GradientDescent:
    def __init__(self, learningRate=0.01, maxIterations=1000, tolerance=1e-6, verbose=False, lambda_=0.0):
        self.learningRate = learningRate
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.adaptive_lr = True
        self.lambda_ = lambda_
        self.min_lr = 1e-10
    
    def getMaxIterations(self):
        return self.maxIterations
        
    def setLambda(self, lambda_):
        self.lambda_ = lambda_

    def optimize(self, X, y, bias=0.0, weights=None):
        X = np.array(X)
        y = np.array(y)
        m = len(y)
        
        if weights is None:
            if X.shape[1] == 1:
                weights = 0.0
            else:
                weights = np.zeros(X.shape[1])
        
        costHistory = []
        cost = self.computeCost(X, y, weights, bias)
        costHistory.append(cost)
        
        for i in range(self.maxIterations):
            old_weights = weights.copy() if isinstance(weights, np.ndarray) else weights
            old_bias = bias
            
            # Compute gradients
            if X.shape[1] == 1:
                X_flat = X.flatten()
                predictions = bias + X_flat * weights
                errors = predictions - y
                
                dw = (1/m) * np.sum(errors * X_flat)
            else:
                predictions = np.dot(X, weights) + bias
                errors = predictions - y
                
                dw = (1/m) * np.dot(X.T, errors)
            
            # Common gradient for bias
            db = (1/m) * np.sum(errors)
            
            # Add regularization term if needed
            if self.lambda_ != 0:
                if np.isscalar(weights):
                    dw += (self.lambda_ / m) * weights
                else:
                    dw += (self.lambda_ / m) * weights
            
            # Update parameters
            weights = weights - self.learningRate * dw
            bias = bias - self.learningRate * db
            
            # Calculate new cost
            new_cost = self.computeCost(X, y, weights, bias)
            costHistory.append(new_cost)
            
            # Check for convergence
            if abs(new_cost - cost) < self.tolerance:
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                break
                
            cost = new_cost
            
        return bias, weights, costHistory
    
    def computeCost(self, X, y, weights, bias):
        m = len(y)
        
        if np.isscalar(weights):
            X_flat = X.flatten()
            predictions = bias + X_flat * weights
        else:
            predictions = np.dot(X, weights) + bias

        regularizationTerm = 0.0
        if self.lambda_ != 0:
            if np.isscalar(weights):
                regularizationTerm = self.lambda_ * (weights ** 2)
            else:
                regularizationTerm = self.lambda_ * np.sum(np.square(weights))

        cost = (1/(2*m)) * np.sum(np.square(y - predictions)) + regularizationTerm
        return cost

class CoordinateDescent:
    def __init__(self, maxIterations = 1000, tolerance = 1e-6, verbose = True, lambda_ = 1.0):
        self.maxIterations = maxIterations
        self.tolerance = tolerance
        self.verbose = verbose
        self.lambda_ = lambda_

    def getMaxIterations(self):
        return self.maxIterations

    def setMaxIterations(self, maxIterations):
        self.maxIterations = maxIterations

    def getTolerance(self):
        return self.tolerance

    def setTolerance(self, tolerance):
        self.tolerance = tolerance

    def getVerbose(self):
        return self.verbose

    def setVerbose(self, verbose):
        self.verbose = verbose

    def getLambda(self):
        return self.lambda_

    def setLambda(self, lambda_):
        self.lambda_ = lambda_

    def optimize(self, X, y):
        X = np.array(X)
        y = np.array(y)
        costHistory = []

        nSamples = X.shape[0]
        mFeatures = X.shape[1]
        
        theta = np.zeros(mFeatures)
        bias = np.mean(y)
        yPred = np.ones(nSamples) * bias

        corrWithTarget = np.abs(X.T @ (y - bias)) / nSamples
        featureOrder = np.argsort(-corrWithTarget)

        cost = self.computeCost(X, y, theta, bias)
        costHistory.append(cost)
        
        costIncreases = 0
        activeSet = set()

        for i in range(self.maxIterations):
            thetaOld = theta.copy()
            maxChange = 0

            activeFeatures = list(activeSet)
            if activeFeatures:
                for j in activeFeatures:
                    oldVal = theta[j]
                    theta[j] = self.computeCoordinate(X, y, theta, j, yPred)

                    if theta[j] != oldVal:
                        yPred += X[:, j] * (theta[j] - oldVal)
                        maxChange = max(maxChange, abs(theta[j] - oldVal))

                    if theta[j] != 0 and j not in activeSet:
                        activeSet.add(j)
                    elif theta[j] == 0 and j in activeSet:
                        activeSet.remove(j)

            if i % 5 == 0 or not activeFeatures:
                for j in featureOrder:
                    if j not in activeSet:
                        oldVal = theta[j]
                        theta[j] = self.computeCoordinate(X, y, theta, j, yPred)

                        if theta[j] != oldVal:
                            yPred += X[:, j] * (theta[j] - oldVal)
                            maxChange = max(maxChange, abs(theta[j] - oldVal))

                        if theta[j] != 0:
                            activeSet.add(j)
            

            newCost = self.computeCost(X, y, theta, bias)
            costHistory.append(newCost)

            if newCost > cost:
                costIncreases += 1
                if costIncreases >= 5:
                    if self.verbose:
                        print(f"Cost increased for {costIncreases} consecutive iterations. Stopping optimization.")
                    break
            else:
                costIncreases = 0

            if self.checkConvergence(cost, newCost, theta, thetaOld):
                if self.verbose:
                    print(f"Converged after {i + 1} iterations")
                break

            cost = newCost


        return theta, bias, costHistory


    def computeCoordinate(self, X, y, params, j, predictions=None):
        N = X.shape[0]
    
        xj = X[:, j]

        squaredSumXJ = np.sum(xj * xj)
        if squaredSumXJ == 0:
            return 0.0
        
        thetaJOld = params[j]
        residuals = y - predictions + xj * thetaJOld
        
        
        # Calculate correlation of feature j with partial residuals
        rho = np.dot(xj, residuals) / N
        
        threshold = self.lambda_ / N
        if abs(rho) <= threshold:
            newParam = 0.0
        else:
            newParam = (rho - np.sign(rho) * threshold) / (squaredSumXJ / N)
            
        return newParam


    def computeCost(self, X, y, theta, bias):
        try:
            N = X.shape[0]
            predictions = y - (X @ theta + bias)
            mse = np.sum(predictions ** 2) / (2 * N)
            l1Penalty = self.lambda_ * np.sum(np.abs(theta)) / N
            cost = mse + l1Penalty
            return cost
        except Exception as e:
            print(f"Error in computeCost: {e}")
            return float('inf')
        
        
    def checkConvergence(self, oldCost, newCost, theta, thetaOld):
        if oldCost == 0:
            objImprovement = abs(newCost)
        else:
            objImprovement = abs((newCost - oldCost) / oldCost)
        
        paramChange = np.max(np.abs(theta - thetaOld))

        return (objImprovement < self.tolerance and 
                paramChange < self.tolerance)

# Define model classes directly in app.py
class UnivariateLinearModel(BaseModel):
    def __init__(self, learningRate=0.01, maxIterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(learningRate=learningRate, maxIterations=maxIterations)
        self.bias = None
        self.weights = None
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None

    def fit(self, X, y):
        self.is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        self.X_columns = X.columns if isinstance(X, pd.DataFrame) else None
        
        X, y = self.validateData(X, y)
        
        if self.normalize:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            X_normalized = self.normalizer.fitTransform(X)
            y_normalized = (y - self.y_mean) / self.y_std
        else:
            X_normalized = X
            y_normalized = y
        
        if X_normalized.shape[1] != 1:
            raise ValueError("SimpleLinearModel expects exactly one feature (univariate regression)")
        
        # Initialize parameters with better starting values
        # For univariate regression, we can directly compute the coefficients
        x_mean = np.mean(X_normalized)
        y_mean = np.mean(y_normalized)
        numerator = np.sum((X_normalized.flatten() - x_mean) * (y_normalized - y_mean))
        denominator = np.sum((X_normalized.flatten() - x_mean) ** 2)
        
        if denominator != 0:
            initialWeights = numerator / denominator
        else:
            initialWeights = 0.0
        
        initialBias = y_mean - initialWeights * x_mean
        
        if isinstance(initialWeights, np.ndarray):
            initialWeights = float(initialWeights.item()) if initialWeights.size == 1 else float(initialWeights[0])
        
        self.bias, weights_array, costHistory = self.optimizer.optimize(
            X_normalized, 
            y_normalized, 
            initialBias, 
            initialWeights
        )
        
        self.weights = float(weights_array) if isinstance(weights_array, np.ndarray) else weights_array
        self.isFitted = True
        self.costHistory = costHistory

        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")

        isPandasInput = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        
        originalIndex = X.index if isPandasInput else None        
        X, _ = self.validateData(X)
        
        if self.normalize:
            X = self.normalizer.transform(X)
        
        X_flat = X.flatten()
        
        predictions = self.bias + X_flat * self.weights
        
        if self.normalize and hasattr(self, 'y_mean') and hasattr(self, 'y_std'):
            predictions = predictions * self.y_std + self.y_mean
        
        if isPandasInput and originalIndex is not None:
            return pd.Series(predictions, index=originalIndex)
        
        return predictions

class MultivariateLinearModel(BaseModel):
    def __init__(self, learningRate=0.01, maxIterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(
            learningRate=learningRate, 
            maxIterations=maxIterations,
            tolerance=1e-8,
            verbose=False
        )
        self.bias = None
        self.weights = None
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None
    
    def fit(self, X, y, verbose=True):
        self.is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        self.X_columns = X.columns if isinstance(X, pd.DataFrame) else None

        X, y = self.validateData(X, y)

        if self.normalize:
            X_normalized = self.normalizer.fitTransform(X)
        else:
            X_normalized = X

        # Initialize weights with small random values instead of zeros
        # This helps avoid getting stuck in bad local minima
        self.weights = np.random.randn(X_normalized.shape[1]) * 0.01
        self.bias = 0.0
        
        self.bias, self.weights, self.costHistory = self.optimizer.optimize(X_normalized, y, self.bias, self.weights)
        self.isFitted = True
        return self
    
    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise Exception("Model has not been trained yet.")
        
        isPandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        
        if isPandas:
            originalIndex = X.index
        
        X, _ = self.validateData(X)
        
        if self.normalize and self.normalizer is not None:
            X = self.normalizer.transform(X)
        
        predictions = np.dot(X, self.weights) + self.bias
        predictions = predictions.flatten()
        
        if isPandas:
            # Make sure the index has the same length as predictions
            if len(originalIndex) != len(predictions):
                return pd.Series(predictions)
            else:
                return pd.Series(predictions, index=originalIndex)
        
        return predictions

class RidgeRegression(MultivariateLinearModel):
    def __init__(self, learningRate=0.01, maxIterations=1000, normalize=True, verbose=False, lambda_=1.0):
        super().__init__(learningRate=learningRate, maxIterations=maxIterations, normalize=normalize)
        self.lambda_ = lambda_
        self.verbose = verbose
        self.optimizer = GradientDescent(
            learningRate=learningRate,
            maxIterations=maxIterations,
            lambda_=self.lambda_,
            tolerance=1e-8,
            verbose=verbose
        )

    def setLambda(self, lambda_, numSamples):
        self.lambda_ = lambda_ / numSamples
        self.optimizer.setLambda(self.lambda_)

    def getLambda(self):
        return self.lambda_
    
    def fit(self, X, y, verbose=True):
        self.is_pandas = isinstance(X, pd.DataFrame) or isinstance(X, pd.Series)
        self.X_columns = X.columns if isinstance(X, pd.DataFrame) else None

        X, y = self.validateData(X, y)

        if self.normalize:
            X_normalized = self.normalizer.fitTransform(X)
        else:
            X_normalized = X

        # Initial coefficients (one for each feature) and intercept
        # Initialize with least squares solution
        XTX = np.dot(X_normalized.T, X_normalized)
        XTy = np.dot(X_normalized.T, y)
        self.weights = np.linalg.solve(XTX + self.lambda_ * np.eye(X_normalized.shape[1]), XTy)
        self.bias = 0.0

        self.bias, self.weights, costHistory = self.optimizer.optimize(X_normalized, y, self.bias, self.weights)
        
        if self.verbose:
            print(f"Model trained with coefficients: {self.weights} and intercept: {self.bias}")
            print(f"Initial cost: {costHistory[0]}")
            print(f"Final cost: {costHistory[-1]}")

        self.isFitted = True
        self.costHistory = costHistory
        return self

class LassoRegression(MultivariateLinearModel):
    def __init__(self, learning_rate=0.01, max_iterations=1000, normalize=True, verbose=False, lambda_=1.0):
        super().__init__(learningRate=learning_rate, maxIterations=max_iterations, normalize=normalize)
        self.lambda_ = lambda_
        self.verbose = verbose
        # Simplified implementation - using gradient descent
        self.optimizer = GradientDescent(
            learningRate=learning_rate,
            maxIterations=max_iterations,
            lambda_=self.lambda_,
            tolerance=1e-8,
            verbose=verbose
        )

    def setLambda(self, lambda_):
        self.lambda_ = lambda_
        self.optimizer.setLambda(lambda_)

    def getLambda(self):
        return self.lambda_

def trainTestSplitData(df, targetCol, testSize=0.2):
    """
    Split a dataset into training and testing sets.

    Parameters:
    -----------
    df: DataFrame, the dataset to split
    targetCol: str, name of the target column
    testSize: float, proportion of the dataset to include in the test split

    Returns:
    --------
    X_train: DataFrame, training data
    X_test: DataFrame, testing data
    y_train: Series, training target values
    y_test: Series, testing target values
    """
    X = df.drop(targetCol, axis=1)
    y = df[targetCol]

    # Calculate the number of rows for the training set
    trainRows = int((1 - testSize) * len(df))

    # Create random indices for shuffling
    indices = np.random.permutation(len(df))

    # Split the data using random indices
    X_train, X_test = X.iloc[indices[:trainRows]], X.iloc[indices[trainRows:]]
    y_train, y_test = y.iloc[indices[:trainRows]], y.iloc[indices[trainRows:]]

    return X_train, X_test, y_train, y_test

# ================== Streamlit App ==================

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
    allDatasets = getAvailableDatasets()
    
    if model_type == "univariate":
        return [dataset for dataset in allDatasets if "univariate" in dataset.lower()]
    elif model_type == "compare_all":
        return [dataset for dataset in allDatasets if "multivariate" in dataset.lower()]
    else:
        return [dataset for dataset in allDatasets if "multivariate" in dataset.lower()]

def trainAllModels(X_train, y_train, X_test, y_test, featureNames, datasetName, learningRate, maxIterations, lambdaParam):
    """Train all model types on the same dataset and return their performance metrics"""
    results = {}
    normalizers = {}
    models = {}
    trainingTimes = {}
    
    # Identify the univariate feature to use based on dataset name
    univariateFeatureIndex = 0  # Default to first feature
    univariateFeatureName = featureNames[0]
    
    if "student" in datasetName.lower():
        univariateFeature = "Previous Scores"
        if univariateFeature in featureNames:
            univariateFeatureIndex = featureNames.index(univariateFeature)
            univariateFeatureName = univariateFeature
    elif "car" in datasetName.lower():
        univariateFeature = "max_power (in bph)"
        if univariateFeature in featureNames:
            univariateFeatureIndex = featureNames.index(univariateFeature)
            univariateFeatureName = univariateFeature
    elif "housing" in datasetName.lower():
        univariateFeature = "median_income"
        if univariateFeature in featureNames:
            univariateFeatureIndex = featureNames.index(univariateFeature)
            univariateFeatureName = univariateFeature
    
    model_types = ["univariate", "multivariate", "ridge", "lasso"]
    
    for modelType in model_types:
        try:
            normalizer = FeatureNormalizer()
            
            if modelType == 'univariate':
                # Extract only the selected univariate feature
                X_trainUnivariate = X_train[:, [univariateFeatureIndex]]
                X_testUnivariate = X_test[:, [univariateFeatureIndex]]
                
                normalizer.fit(X_trainUnivariate)
                X_trainNormalized = normalizer.transform(X_trainUnivariate)
                X_testNormalized = normalizer.transform(X_testUnivariate)
                
                model = UnivariateLinearModel(
                    learningRate=learningRate,
                    maxIterations=maxIterations
                )
            elif modelType == 'multivariate':
                normalizer.fit(X_train)
                X_trainNormalized = normalizer.transform(X_train)
                X_testNormalized = normalizer.transform(X_test)
                
                model = MultivariateLinearModel(
                    learningRate=learningRate,
                    maxIterations=maxIterations
                )
            elif modelType == 'ridge':
                normalizer.fit(X_train)
                X_trainNormalized = normalizer.transform(X_train)
                X_testNormalized = normalizer.transform(X_test)
                
                model = RidgeRegression(
                    learningRate=learningRate,
                    maxIterations=maxIterations,
                    lambda_=lambdaParam
                )
            elif modelType == 'lasso':
                normalizer.fit(X_train)
                X_trainNormalized = normalizer.transform(X_train)
                X_testNormalized = normalizer.transform(X_test)
                
                model = LassoRegression(
                    learning_rate=learningRate,
                    max_iterations=maxIterations,
                    lambda_=lambdaParam
                )
            
            # Train the model
            startTime = time.time()
            model.fit(X_trainNormalized, y_train)
            trainingTime = time.time() - startTime
            
            # Calculate scores
            if modelType == 'univariate':
                trainScore = model.score(X_trainNormalized, y_train)
                testScore = model.score(X_testNormalized, y_test)
            else:
                trainScore = model.score(X_trainNormalized, y_train)
                testScore = model.score(X_testNormalized, y_test)
            
            # Store results
            normalizers[modelType] = normalizer
            models[modelType] = model
            trainingTimes[modelType] = trainingTime
            
            results[modelType] = {
                'train_score': trainScore,
                'test_score': testScore,
                'training_time': trainingTime
            }
            
            if modelType == 'univariate':
                results[modelType]['feature_used'] = univariateFeatureName
            
        except Exception as e:
            st.error(f"Error training {modelType} model: {str(e)}")
            results[modelType] = {
                'train_score': 0,
                'test_score': 0,
                'training_time': 0,
                'error': str(e)
            }
    
    return results, normalizers, models, trainingTimes

st.title("Linear Regression Models")

resetSessionState()

with st.sidebar:
    st.header("Model Configuration")
    
    modelType = st.selectbox(
        "Model Type",
        options=["univariate", "multivariate", "ridge", "lasso", "compare_all"],
        format_func=lambda x: {
            "univariate": "Univariate Linear Regression",
            "multivariate": "Multivariate Linear Regression",
            "ridge": "Ridge Regression",
            "lasso": "Lasso Regression",
            "compare_all": "Compare All Models"
        }[x]
    )
    
    learningRate = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001, format="%.4f")
    maxIterations = st.number_input("Max Iterations", min_value=10, max_value=10000, value=1000, step=100)
    
    if modelType in ["ridge", "lasso", "compare_all"]:
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
            
            featureCols = data.columns[:-1].tolist()
            targetCol = data.columns[-1]
            
            st.session_state.data['feature_names'] = featureCols
            st.session_state.data['target_name'] = targetCol
            
            X = data[featureCols].values
            y = data[targetCol].values
            
            st.session_state.data['X'] = X
            st.session_state.data['y'] = y
            
            X_train, X_test, y_train, y_test = trainTestSplitData(data, targetCol, testSize=testSize)
            
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
                X_test = st.session_state.data['X_test']
                y_test = st.session_state.data['y_test']
                
                if modelType == 'compare_all':
                    # Train all models at once
                    results, normalizers, models, trainingTimes = trainAllModels(
                        X_train, y_train, X_test, y_test,
                        st.session_state.data['feature_names'],
                        selectedDataset,
                        learningRate, maxIterations, lambdaParam
                    )
                    
                    st.session_state.comparison_results = results
                    st.session_state.models = models
                    st.session_state.normalizers = normalizers
                    st.session_state.training_times = trainingTimes
                    
                    st.success(f"All models trained successfully")
                else:
                    # Train single model
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
                            learning_rate=learningRate,
                            max_iterations=maxIterations,
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

    if modelType == 'compare_all' and 'comparison_results' in st.session_state:
        # Display comparison of all models
        st.header("Model Comparison")
        
        results = st.session_state.comparison_results
        
        # Create a comparison dataframe
        comparisonData = []
        for modelName, metrics in results.items():
            comparisonData.append({
                'Model': {
                    'univariate': 'Univariate Linear Regression',
                    'multivariate': 'Multivariate Linear Regression',
                    'ridge': 'Ridge Regression',
                    'lasso': 'Lasso Regression'
                }[modelName],
                'Training R²': metrics['train_score'],
                'Test R²': metrics['test_score'],
                'Training Time (s)': metrics['training_time']
            })
        
        comparisonDF = pd.DataFrame(comparisonData)
        
        st.dataframe(comparisonDF.style.highlight_max(subset=['Test R²'], axis=0))
        
        # If univariate model was trained, show which feature was used
        if 'univariate' in results and 'feature_used' in results['univariate']:
            st.info(f"The univariate model used the '{results['univariate']['feature_used']}' feature.")
        
        # Visualization of model comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = [model['Model'] for model in comparisonData]
        trainingScores = [model['Training R²'] for model in comparisonData]
        testScores = [model['Test R²'] for model in comparisonData]
        
        barWidth = 0.35
        index = np.arange(len(x))
        
        plt.bar(index, trainingScores, barWidth, label='Training R²', color='skyblue')
        plt.bar(index + barWidth, testScores, barWidth, label='Test R²', color='orange')
        
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison')
        plt.xticks(index + barWidth / 2, x, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Display training time comparison
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        trainingTimes = [model['Training Time (s)'] for model in comparisonData]
        
        plt.bar(index, trainingTimes, color='green')
        plt.xlabel('Model')
        plt.ylabel('Training Time (seconds)')
        plt.title('Training Time Comparison')
        plt.xticks(index, x, rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig2)

    elif modelType != 'compare_all' and modelType in st.session_state.models and st.session_state.models[modelType] is not None:
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
                
                if hasattr(model, 'costHistory'):
                    costHistory = model.costHistory
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
                # Create multiple visualizations for multivariate models
                st.subheader("Model Fit Visualization")
                
                # 1. Predicted vs Actual Plot
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                
                ax1.scatter(y_trainPred, y_train, alpha=0.5, label='Training Data')
                ax1.scatter(y_testPred, y_test, alpha=0.5, label='Test Data')
                
                # Calculate the range for the diagonal line
                yAll = np.concatenate([y_train, y_test])
                yAllPred = np.concatenate([y_trainPred, y_testPred])
                minVal = min(np.min(yAll), np.min(yAllPred))
                maxVal = max(np.max(yAll), np.max(yAllPred))
                
                # Plot diagonal line representing perfect predictions
                ax1.plot([minVal, maxVal], [minVal, maxVal], 'r--', label='Perfect Prediction')
                
                ax1.set_xlabel('Predicted Values')
                ax1.set_ylabel('Actual Values')
                ax1.legend()
                ax1.set_title('Predicted vs Actual Values')
                
                st.pyplot(fig1)
                
                # 2. Residuals Plot
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                
                trainResiduals = y_train - y_trainPred
                testResiduals = y_test - y_testPred
                
                ax2.scatter(y_trainPred, trainResiduals, alpha=0.5, label='Training Data')
                ax2.scatter(y_testPred, testResiduals, alpha=0.5, label='Test Data')
                ax2.axhline(y=0, color='r', linestyle='-', label='Perfect Fit')
                
                ax2.set_xlabel('Predicted Values')
                ax2.set_ylabel('Residuals')
                ax2.legend()
                ax2.set_title('Residual Plot (Actual - Predicted)')
                
                st.pyplot(fig2)
                
                # 3. Distribution of Residuals
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                
                sns.histplot(trainResiduals, kde=True, label='Training Residuals', alpha=0.5, ax=ax3)
                sns.histplot(testResiduals, kde=True, label='Test Residuals', alpha=0.5, ax=ax3)
                
                ax3.axvline(x=0, color='r', linestyle='--', label='Zero Residual')
                ax3.set_xlabel('Residual Value')
                ax3.set_ylabel('Frequency')
                ax3.legend()
                ax3.set_title('Distribution of Residuals')
                
                st.pyplot(fig3)
                
                # 4. If 2D plot is possible (when we have just 2 features and they're the most important)
                if len(st.session_state.data['feature_names']) >= 2:
                    # Get the two most important features
                    featureImportance = abs(model.weights.flatten())
                    features = st.session_state.data['feature_names']
                    
                    importanceDF = pd.DataFrame({
                        'Feature': features,
                        'Importance': featureImportance,
                        'Index': range(len(features))
                    })
                    importanceDF = importanceDF.sort_values(by='Importance', ascending=False)
                    
                    if importanceDF.shape[0] >= 2:
                        topFeatures = importanceDF.head(2)
                        feat1Index = topFeatures.iloc[0]['Index']
                        feat2Index = topFeatures.iloc[1]['Index']
                        
                        feat1Name = topFeatures.iloc[0]['Feature']
                        feat2Name = topFeatures.iloc[1]['Feature']
                        
                        fig4, ax4 = plt.subplots(figsize=(10, 8))
                        
                        # Create a meshgrid for the top two features
                        xFeat1 = np.linspace(np.min(X_train[:, feat1Index]), np.max(X_train[:, feat1Index]), 30)
                        xFeat2 = np.linspace(np.min(X_train[:, feat2Index]), np.max(X_train[:, feat2Index]), 30)
                        xx1, xx2 = np.meshgrid(xFeat1, xFeat2)
                        
                        # Create input grid for prediction
                        gridPoints = np.zeros((xx1.ravel().shape[0], X_train.shape[1]))
                        
                        # Use mean values for other features
                        meanValues = np.mean(X_train, axis=0)
                        for i in range(X_train.shape[1]):
                            if i != feat1Index and i != feat2Index:
                                gridPoints[:, i] = meanValues[i]
                        
                        # Set the values for the two chosen features
                        gridPoints[:, feat1Index] = xx1.ravel()
                        gridPoints[:, feat2Index] = xx2.ravel()
                        
                        # Normalize the grid points
                        gridPointsNormalized = normalizer.transform(gridPoints)
                        
                        # Predict and reshape
                        zPred = model.predict(gridPointsNormalized).reshape(xx1.shape)
                        
                        # Plot the contour
                        cs = ax4.contourf(xx1, xx2, zPred, levels=15, cmap='viridis', alpha=0.5)
                        cbar = plt.colorbar(cs, ax=ax4)
                        cbar.set_label(st.session_state.data['target_name'])
                        
                        # Scatter actual data points
                        scatter = ax4.scatter(X_train[:, feat1Index], X_train[:, feat2Index], 
                                             c=y_train, cmap='viridis', edgecolor='k', s=50)
                        
                        ax4.set_xlabel(feat1Name)
                        ax4.set_ylabel(feat2Name)
                        ax4.set_title(f'Model Prediction Surface for Top 2 Features\n(other features set to mean values)')
                        
                        st.pyplot(fig4)
                
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
                    
                    fig5, ax5 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=importanceDF, ax=ax5)
                    ax5.set_title('Feature Importance')
                    st.pyplot(fig5)
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
    5. **Compare All Models**: Train all models simultaneously on the same dataset and compare their performance
    
    Choose a dataset and model type from the sidebar to get started.
    """)
