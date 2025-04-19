import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.LassoRegression import LassoRegression
from LinearRegression.utils.DataLoader import loadDatasetFromCSV
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
from LinearRegression.preprocessing.Normalization import FeatureNormalizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

studentData = loadDatasetFromCSV("../datasets/multivariateStudentData.csv")
housingData = loadDatasetFromCSV("../datasets/multivariateHousingData.csv")
carPriceData = loadDatasetFromCSV("../datasets/multivariateCarPricesData.csv")

maxIterations = 5000
k_folds = 5  # Number of folds for cross-validation

datasets = [studentData, housingData, carPriceData]
targets = ["Performance Index", "median_house_value", "selling_price"]

plt.figure(figsize=(15, 15))

# Lambda values to try
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1.0]

for i in range(len(datasets)):
    dataset = datasets[i]
    targetCol = targets[i]

    print(f"\n{'='*50}")
    print(f"DATASET {i+1}: {targetCol}")
    print(f"{'='*50}")
    print(f"Shape: {dataset.shape}")
    print(f"Number of features: {dataset.shape[1] - 1}")
    print(f"Target range: {dataset[targetCol].min()} - {dataset[targetCol].max()}")
    
    X = dataset.drop(targetCol, axis=1)
    y = dataset[targetCol]
    
    # Normalize target variable
    y_normalizer = FeatureNormalizer()
    y_normalized = y_normalizer.fitTransform(y.values.reshape(-1, 1)).flatten()
    
    print(f"\n1. Tuning lambda with cross-validation")
    print(f"{'-'*30}")
    
    bestLambda = None
    bestCV_score = float('-inf')

    model = LassoRegression(
        max_iterations=maxIterations,
        normalize=True
    )
    
    for lambda_ in lambda_values:
        print(f"\nTrying lambda = {lambda_}")
        
        model.setLambda(lambda_)
        
        score = model.crossValidation(X, y_normalized, k=k_folds)
        meanCV_score = np.mean(score)
        
        print(f"Mean CV R² score: {meanCV_score:.4f}")
        
        if meanCV_score > bestCV_score:
            bestCV_score = meanCV_score
            bestLambda = lambda_
    
    print(f"\nBest lambda: {bestLambda}")
    print(f"Best CV R² score: {bestCV_score:.4f}")
    
    print(f"\n2. TEST SET RESULTS")
    print(f"{'-'*30}")
    
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, targetCol)
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    yTrainNormalized = y_normalizer.transform(y_train.values.reshape(-1, 1)).flatten()
    yTestNormalized = y_normalizer.transform(y_test.values.reshape(-1, 1)).flatten()

    
    model.setLambda(bestLambda)
    model.fit(X_train, yTrainNormalized, verbose=False)

    trainScore = model.score(X_train, yTrainNormalized)
    testScore = model.score(X_test, yTestNormalized)
    
    print(f"Model R^2 score on training data: {trainScore:.4f}")
    print(f"Model R^2 score on test data: {testScore:.4f}")

    
    print("---------------------------------------------------------")