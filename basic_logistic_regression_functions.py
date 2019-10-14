import numpy as np

def predict(X,theta):
  # calculates the prediction h_theta(x) for input(s) x contained in array X
  return np.sign(X.dot(theta.T))

def computeCost(X, y, theta):
  # function calculates the cost J(theta) and returns its value
  exponent_power = np.multiply(-y,X.dot(theta.T))
  return np.divide(np.sum(np.log(1 + np.exp(exponent_power))), len(X))

def computeGradient(X,y,theta):
  # calculate the gradient of J(theta) and return its value

  exponent = np.exp(np.multiply(-y, X.dot(theta.T)))
  # rhs represents the right hand side of the
  # computeGradient equation
  rhs = np.divide(exponent, 1 + exponent)
  return np.divide(np.multiply(-y,rhs).dot(X),len(X))

def addQuadraticFeature(X):
  # Given feature vector [x_1,x_2] as input, extend this to
  # [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature4
  cols = np.hsplit(X,2)
  first_elems = cols[0] ** 2
  return np.append(X, first_elems, axis = 1)

def computeScore(X,y,preds):
  # for training data X,y it calculates the number of correct predictions made by the model
  return np.sum(y==preds)