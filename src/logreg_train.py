import logging
# logging.basicConfig(level=logging.DEBUG)

import numpy as np

from describe import checkInput, readCSV

def sigmoid(z):
	return 1 / (1 + np.exp(-z)) # Logistic sigmoid function ( 1 / (1 + e^-z) ) e = Euler's number (~ 2.71828) 

def hypothesis(X, theta):
	return sigmoid(np.dot(X, theta)) # Hypothesis function hθ(x) = sigmoid(θ.T * x)



if "__main__" == __name__:
	data = readCSV(checkInput())

	numeric_columns = [col for col in data.columns if (data[col].dtype == 'float64' or data[col].dtype == 'int64') and not data[col].isna().all() and col != 'Index']
	if len(numeric_columns) == 0:
		logging.error("No numeric columns found in the dataset to plot histograms.")
		exit()

	
	