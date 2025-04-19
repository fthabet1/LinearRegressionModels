import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.MultivariateLinearModel import MultivariateLinearModel
from LinearRegression.utils.DataLoader import loadDatasetFromCSV
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
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

for i in range(len(datasets)):
    dataset = datasets[i]
    targetCol = targets[i]

    print(f"Dataset {i+1}: {targetCol}")
    print(f"Shape: {dataset.shape}")
    print(f"Number of features: {dataset.shape[1] - 1}")
    print(f"Target range: {dataset[targetCol].min()} - {dataset[targetCol].max()}")
    
    X = dataset.drop(targetCol, axis=1)
    y = dataset[targetCol]
    
    model = MultivariateLinearModel(
        learningRate=0.05,
        maxIterations=5000,
        normalize=True
    )
    
    scores = model.crossValidation(X, y, k=k_folds)
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, targetCol)

    model.fit(X_train, y_train, verbose=False)
    trainScore = model.score(X_train, y_train)
    testScore = model.score(X_test, y_test)
    
    print(f"Model R^2 score on training data: {trainScore:.4f}")
    print(f"Model R^2 score on test data: {testScore:.4f}")

    
    print("---------------------------------------------------------")