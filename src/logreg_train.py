import logging
# logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd

from describe import checkInput, readCSV

pd.set_option('display.float_format', '{:.17g}'.format)
pd.set_option('display.max_columns', None)



def sigmoid(z):
	return 1 / (1 + np.exp(-z)) # Logistic sigmoid function ( 1 / (1 + e^-z) ) e = Euler's number (~ 2.71828) 

def hypothesis(X, theta):
	return sigmoid(np.dot(X, theta)) # Hypothesis function hθ(x) = sigmoid(θ.T * x)

def normalizeFeatures(X):
	describer = readCSV("describe_output.csv")
	describer.drop(['Index', "Unnamed: 0"], axis=1, inplace=True)
	means = describer.iloc[4].values
	std_devs = describer.iloc[5].values

	X_norm = (X - means) / std_devs
	return X_norm

def ravenclawMapping(y_column):
	mapping = {'Gryffindor': 0, 'Hufflepuff': 0, 'Ravenclaw': 1, 'Slytherin': 0}
	return y_column.map(mapping).to_numpy().reshape(-1, 1)

def ravenclawLogReg(X, y, alpha=0.01, num_iterations=1000):
	y = ravenclawMapping(y)
	m, n = X.shape
	theta = np.zeros((n, 1))

	cost_history = []


	for i in range(num_iterations):

		h = hypothesis(X, theta)
		gradient = np.dot(X.T, (h - y)) / m
		theta = theta - ( alpha * gradient )

		if i % 50 == 0:
			cost = (-1 / m) * ( np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)) )
			cost_history.append(cost)

	

	return theta, cost_history
	
	

if "__main__" == __name__:
	data = readCSV(checkInput())

	numeric_columns = [col for col in data.columns if (data[col].dtype == 'float64' or data[col].dtype == 'int64') and not data[col].isna().all() and col != 'Index']
	if len(numeric_columns) == 0:
		logging.error("No numeric columns found in the dataset to plot histograms.")
		exit()

	normalized_data = normalizeFeatures(data[numeric_columns].copy())
	normalized_data.fillna(0, inplace=True)
	normalized_data.insert(0, 'Bias', 1)

	X = normalized_data.copy().to_numpy()
	y_column = data['Hogwarts House']

	theta, cost_history = ravenclawLogReg(X, y_column, alpha=0.01, num_iterations=1000)


	test = [68570.0,-387.74483245021133,-4.604619969975151,3.8774483245021134,-6.945,-419.16429373366526,403.71291151908974,1.72055017703211,1058.5804798295928,7.637557962209646,-0.4045276627737053,-253.98429,-54.3]
	test2 = [73385.0,-415.0263289492602,7.254609476514043,4.150263289492602,6.916,489.1024503769474,667.2904112153512,7.920045613054778,1058.228762712228,9.549079995699305,-1.6593503778036771,-227.92076,-28.93]
	test_norm = normalizeFeatures(pd.DataFrame([test2], columns=numeric_columns))
	test_norm.insert(0, 'Bias', 1)
	prediction = hypothesis(test_norm.to_numpy(), theta)
	print("Ravenclaw prediction for test data point:", prediction[0][0])