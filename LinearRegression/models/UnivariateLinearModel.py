import pandas as pd
import numpy as np
import sys
import os

# Add root to path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent
from LinearRegression.preprocessing.Normalization import FeatureNormalizer

class UnivariateLinearModel(BaseModel):

    def __init__(self, learningRate=0.01, maxIterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(learningRate=learningRate, maxIterations=maxIterations)
        self.bias = None
        self.weights = None
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None

    def fit(self, X, y):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x 1 array of N samples and 1 feature)
        y: Array of target values (N samples)

        Returns:
        --------
        self: returns an instance of self
        """
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
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X: Array of data to make predictions on (N x 1 array of N samples and 1 feature)

        Returns:
        --------
        yPred: Array of predicted values (N predictions)
        """
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