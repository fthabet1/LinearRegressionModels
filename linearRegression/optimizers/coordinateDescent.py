import numpy as np

class CoordinateDescent:

    def __init__(self, maxIterations = 1000, tolerance = 1e-6, verbose = True, lambda_ = 1.0):
        """
        Initialize the coordinate descent optimizer.
        
        Parameters:
        -----------
        learningRate: float, the step size to take in the direction of the gradient
        maxIterations: int, the maximum number of iterations to run
        tolerance: float, convergence threshold for stopping
        verbose: bool, whether to print out the loss at each iteration

        Returns:
        --------
        self: an instance of self
        """

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
        """
        Optimize the model parameters using gradient descent.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        initalParams: Array of initial model parameters

        Returns:
        --------
        params: Array of optimized model parameters
        costHistory: Array of cost function history
        """

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
        """
        Compute the coordinate descent update for a single parameter.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        params: Array of model parameters
        j: int, the index of the parameter to update
        predictions: Array of current predictions (optional, for efficiency)

        Returns:
        --------
        newParam: float, the updated parameter value
        """
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
        """
        Calculate the cost function.

        Parameters:
        -----------
        X: Array of training data (N x M array of N samples and M features)
        y: Array of target values (N samples)
        params: Array of model parameters

        Returns:
        --------
        cost: float, the cost of the model
        """
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
        """
        Check convergence for Lasso regression.
        
        Parameters:
        -----------
        oldCost: float, previous iteration's cost
        newCost: float, current iteration's cost
        theta: array, current parameters
        thetaOld: array, previous parameters
        
        Returns:
        --------
        bool: True if converged, False otherwise
        """
        # Check relative improvement in objective
        if oldCost == 0:
            objImprovement = abs(newCost)
        else:
            objImprovement = abs((newCost - oldCost) / oldCost)
        
        # Check parameter changes
        paramChange = np.max(np.abs(theta - thetaOld))
        
        # Convergence criteria:
        # 1. Objective improvement is small
        # 2. Parameter changes are small
        # 4. Minimum improvement threshold is met
        return (objImprovement < self.tolerance and 
                paramChange < self.tolerance)