import numpy as np

def predict(X,theta):
  # function calculates the prediction h_theta(x) and returns its value
  return np.sign(X.dot(theta.T))

def computeCost(X, y, theta, lambd):
  # function calculates the cost J(theta) and returns its value

  sum_val = np.sum(np.maximum(0, 1 - np.multiply(y, X.dot(theta.T))))
  # weight // scalar value
  weight = np.multiply(lambd, theta.dot(theta.T))
  return np.divide(sum_val, len(X)) + weight

def computeGradient(X,y,theta,lambd):
  # calculate the gradient of J(theta) and return its value
  subgradient_vals = np.multiply(y, X.dot(theta.T)) <= 1
  sum_val = np.multiply(y,subgradient_vals.astype(np.int)).dot(X)
  rhs = np.divide(sum_val, len(X))
  return np.subtract(np.multiply(2*lambd, theta), rhs)

def addQuadraticFeature(X):
  # Given feature vector [x_1,x_2] as input, extend this to
  # [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature
  cols = np.hsplit(X,2)
  first_elems = cols[0] ** 2
  return np.append(X, first_elems, axis = 1)

def computeScore(X,y,preds):
  # for training data X,y it calculates the number of correct predictions made by the model
  return np.sum(y==preds)