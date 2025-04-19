import pandas as pd
import numpy as np
import sys
import os

# Add root to path to enable absolute imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from LinearRegression.models.BaseModel import BaseModel
from LinearRegression.optimizers.GradientDescent import GradientDescent
from LinearRegression.preprocessing.Normalization import FeatureNormalizer

class MultivariateLinearModel(BaseModel):

    def __init__(self, learningRate=0.01, maxIterations=1000, normalize=True):
        super().__init__()
        self.optimizer = GradientDescent(
            learningRate=learningRate, 
            maxIterations=maxIterations,
            tolerance=1e-8,  # Use a smaller tolerance for better convergence
            verbose=False
        )
        self.bias = None
        self.weights = None
        self.normalize = normalize
        self.normalizer = FeatureNormalizer() if normalize else None
    
    def fit(self, X, y, verbose=True):
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        verbose: Whether to print progress information

        Returns:
        --------
        self: returns an instance of self
        """
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
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X: Array of data to make predictions on (N x M array of N samples and M features)

        Returns:
        --------
        yPred: Array of predicted values (N predictions)
        """
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