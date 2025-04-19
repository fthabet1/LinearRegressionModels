from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):
    
    def __init__(self):
        # These are common attributes among all regression models
        self.weights = None  # Storing model weights
        self.bias = None  # Storing the intercept term
        self.isFitted = False  # Track if the model has been trained or not

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
        """
        Train the model on the provided data.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)

        Returns:
        --------
        self: returns an instance of self
        """
        pass

    @abstractmethod
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
        pass

    def score(self, X, y):
        """
        Calculate the coefficient of determination (R^2) for the model.

        Parameters:
        -----------
        X: Array of test data (N x M array of N samples and M features)
        y: Array of actual values (N samples)

        Returns:
        --------
        score: float, R^2 score
        """
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
        
        # R² sometimes can be negative if the model is worse than predicting the mean
        # In practice, it's usually capped at 0
        return max(0.0, r2) if not np.isnan(r2) else 0.0
    
    def validateData(self, X, y=None):
        """
        Validate and convert input data to proper numpy arrays.
        
        This function should:
        1. Convert X to a numpy array if it isn't already
        2. Ensure X is 2D (n samples, m features)
        3. If y is provided, convert it to a numpy array if it isn't already
        4. Ensure y is 1D (n samples)
        5. Check for missing or infinite values
        6. Verify X and y have the same number of samples

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)

        Returns:
        --------
        X: Numpy array of training data
        y: Numpy array of target values (if provided) 
        """
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

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError("X contains non-numeric data, which is not supported")
        
        if np.isnan(X).any():
            raise ValueError("X contains missing values")

        if np.isinf(X).any():
            raise ValueError("X contains infinite values")

        if y is not None:
            if np.isnan(y).any():
                raise ValueError("y contains missing values")

            if np.isinf(y).any():
                raise ValueError("y contains infinite values")

        return X, y if y is not None else None 