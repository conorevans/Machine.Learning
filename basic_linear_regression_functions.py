import numpy as np

def predict(X,theta):
  # takes m by n matrix X as input and returns an mx1 vector containing the predictions h_theta(x^i) for each row x^i, i=1,...,m in X
  return np.dot(X,theta)

def computeCost(X, y, theta):
  # function calculates the cost J(theta) and return its value
  return (1 / (2 * len(X))) * (np.sum((predict(X,theta) - y) ** 2))

def computeGradient(X,y,theta):
  # function calculate the gradient of J(theta) and returns its value
  return (1 / len(X)) * X.T.dot((predict(X,theta) - y))