import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LinearRegression.models.UnivariateLinearModel import UnivariateLinearModel
from LinearRegression.utils.DataLoader import loadDatasetFromCSV
from LinearRegression.preprocessing.DataSplitter import trainTestSplitData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

salaryData = loadDatasetFromCSV("../datasets/univariateSalaryData.csv")
iceCreamData = loadDatasetFromCSV("../datasets/univariateIceCreamData.csv")
studentData = loadDatasetFromCSV("../datasets/univariateStudentData.csv")

datasets = [salaryData, iceCreamData, studentData]
targets = ["Salary", "Ice Cream Profits", "Final Grade"]
feature_cols = ["YearsExperience", "Temperature", "studytime"]

learningRates = [0.01, 0.001, 0.05]
maxIterations = [5000, 5000, 5000]

plt.figure(figsize=(18, 12))

for i in range(len(datasets)):
    dataset = datasets[i]
    featureCol = feature_cols[i]
    targetCol = targets[i]
    
    print(f"\nDataset {i+1}: {featureCol} vs {targetCol}")
    print(f"Shape: {dataset.shape}")
    print(f"Feature range: {dataset[featureCol].min()} - {dataset[featureCol].max()}")
    print(f"Target range: {dataset[targetCol].min()} - {dataset[targetCol].max()}")
    
    X_train, X_test, y_train, y_test = trainTestSplitData(dataset, targetCol)
    
    model = UnivariateLinearModel(
        learningRate=learningRates[i],
        maxIterations=maxIterations[i],
        normalize=True
    )
    
    scoresCV = model.crossValidation(dataset[featureCol], dataset[targetCol], k=5)

    print(f"Cross-validation R² scores: {[round(score, 4) for score in scoresCV]}")

    model.fit(X_train, y_train)
    trainScore = model.score(X_train, y_train)
    testScore = model.score(X_test, y_test)
    
    print(f"Model R^2 score on training data: {trainScore:.4f}")
    print(f"Model R^2 score on test data: {testScore:.4f}")

    
    print("---------------------------------------------------------")
