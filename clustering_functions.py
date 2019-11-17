import numpy as np

def distance(X,mu):
  # calculate the euclidean distance between numpy arrays X and mu
  return np.sum(np.square(np.subtract(X, mu)), axis=1)
	
def findClosestCentres(X,mu):
  # finds the centre in mu closest to each point in X
  
  m = X.shape[0] # number of points
  k = mu.shape[0] # number of centres
  C = list()
  
  # create list of empty arrays
  # tried with various np.empty permutations but no luck
  for i in range(k):
    C.append([])

  # for each point in X, find the point in mu where the distance
  # is minimum
  for i in range(m):
    # get array of all distances and the min value of the array
    distances = distance(X[i], mu)
    min_val = distances.min()
    # np.where returns a tuple, the first element [0] is the np array
    # abs can only work on one value so we can't do this step above without
    # changing the behaviour of distance as well as this function
    min_index = np.where(abs(min_val) == distances)[0]
    # min index is actually an np.array so 
    # we access the value by doing [0] again
    # and append the current index of X at the index within the array
    # of centres
    C[min_index[0]].append(i)
    
  return C
  
def updateCentres(X,C):
  # updates the centres to be the average of the points closest to it.  
  k = len(C) # k is number of centres
  n = X.shape[1] # n is number of features
  mu = np.zeros((k,n))
  for i in range(len(C)):
    # for each list of centres, np.sum X[list_indices] along the
    # vertical axis i.e. X[i][0] + X[i+1][0]...
    # and divide by the number of centres : len(C[i])
    mu[i] = np.divide(np.sum(X[C[i]], axis=0), len(C[i]))

  return mu