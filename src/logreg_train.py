import logging
# logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from describe import checkInput, readCSV

pd.set_option('display.float_format', '{:.17g}'.format)
pd.set_option('display.max_columns', None)

def sigmoid(z):
	return 1 / (1 + np.exp(-z)) # Logistic sigmoid function ( 1 / (1 + e^-z) ) e = Euler's number (~ 2.71828) 

def hypothesis(X, theta):
	return sigmoid(np.dot(X, theta)) # Hypothesis function hθ(x) = sigmoid(θ.T * x)

def normalizeFeatures(X):
	describer = readCSV("describe_output.csv")
	# describer.drop(['Care of Magical Creatures', 'Arithmancy', 'Index', "Unnamed: 0"], axis=1, inplace=True)
	describer.drop(['Care of Magical Creatures', 'Arithmancy', 'Index', "Unnamed: 0", "Potions", "Astronomy"], axis=1, inplace=True)
	means = describer.iloc[4].values
	std_devs = describer.iloc[5].values

	X_norm = (X - means) / std_devs
	return X_norm

def yMapping(y_column, id):
	match id:
		case 0:
			mapping = {'Gryffindor': 1, 'Hufflepuff': 0, 'Ravenclaw': 0, 'Slytherin': 0}
		case 1:
			mapping = {'Gryffindor': 0, 'Hufflepuff': 1, 'Ravenclaw': 0, 'Slytherin': 0}
		case 2:
			mapping = {'Gryffindor': 0, 'Hufflepuff': 0, 'Ravenclaw': 1, 'Slytherin': 0}
		case 3:
			mapping = {'Gryffindor': 0, 'Hufflepuff': 0, 'Ravenclaw': 0, 'Slytherin': 1}
		case _:
			logging.error("Invalid id for yMapping function.")
			exit()
	return y_column.map(mapping).to_numpy().reshape(-1, 1)

def gradientDescent(X, y, id, alpha=0.01, num_iterations=10000):
	y = yMapping(y, id)
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


def stochasticGradientDescent(X, y, id, alpha=0.01, epochs=10):
	y = yMapping(y, id)
	m, n = X.shape
	theta = np.zeros((n, 1))

	cost_history = []

	for epoch in range(epochs):

		indices = np.random.permutation(m)
		X_shuffled = X[indices]
		y_shuffled = y[indices]

		for i in range(m):

			xi = X_shuffled[i].reshape(1, -1)
			yi = y_shuffled[i].reshape(1, 1)

			h = hypothesis(xi, theta)
			gradient = np.dot(xi.T, (h - yi))
			theta = theta - ( alpha * gradient )

			if (i % 50 == 0):
				cost = (-1) * ( yi.T * np.log(h) + (1 - yi).T * np.log(1 - h) )
				cost_history.append(cost)

	return theta, cost_history

def minibatchGradienDescent(X, y, id, alpha=0.01, epochs=10, batch_size=32):
	y = yMapping(y, id)
	m, n = X.shape
	theta = np.zeros((n, 1))

	cost_history = []

	for epoch in range(epochs):

		indices = np.random.permutation(m)
		X_shuffled = X[indices]
		y_shuffled = y[indices]

		for i in range(0, m, batch_size):

			X_batch = X_shuffled[i:i+batch_size]
			y_batch = y_shuffled[i:i+batch_size]

			m_batch = X_batch.shape[0]

			h = hypothesis(X_batch, theta)
			gradient = np.dot(X_batch.T, (h - y_batch)) / m_batch
			theta = theta - ( alpha * gradient )

			if (i % 50 == 0):
				cost = (-1 / m_batch) * ( np.dot(y_batch.T, np.log(h)) + np.dot((1 - y_batch).T, np.log(1 - h)) )
				cost_history.append(cost)

	return theta, cost_history



def plotCostHistory(cost_history):

	fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(18, 9))
	axes = axes.flatten()
	methods = ['Gradient Descent', 'Stochastic Gradient Descent']
	houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
	for i in range(4):
		for j in range(2):
			method_name = methods[j]
			history_index = i * 2 + j
			axes[history_index].plot(range(len(cost_history[history_index])), [cost.flatten()[0] for cost in cost_history[history_index]], color='blue')
			axes[history_index].set_title(f'Cost History - {method_name} for {houses[i]}')
			axes[history_index].set_xlabel('Iterations')
			axes[history_index].set_ylabel('Cost')
			axes[history_index].grid()
	plt.tight_layout()
	plt.show()

if "__main__" == __name__:
	data = readCSV(checkInput("logreg_train"))

	# drop_columns = ['Index', 'Care of Magical Creatures', 'Arithmancy']
	drop_columns = ['Index', 'Care of Magical Creatures', 'Arithmancy', "Potions", "Astronomy"]

	numeric_columns = [col for col in data.columns if (data[col].dtype == 'float64' or data[col].dtype == 'int64') and not data[col].isna().all() and col not in drop_columns]
	if len(numeric_columns) == 0:
		logging.error("No numeric columns found in the dataset to plot histograms.")
		exit()

	normalized_data = normalizeFeatures(data[numeric_columns].copy())
	normalized_data.fillna(0, inplace=True)
	normalized_data.insert(0, 'Bias', 1)

	X = normalized_data.copy().to_numpy()
	y_column = data['Hogwarts House']

	gd_theta_df = pd.DataFrame()
	sgd_theta_df = pd.DataFrame()

	final_cost_history = []

	for i in range(4):
		theta, cost_history = gradientDescent(X, y_column, id=i, alpha=0.01, num_iterations=10000)
		final_cost_history.append(cost_history)
		print("\033[30mGD Cost : " + str(cost_history[-1].flatten()[0]) + "\033[0m")
		sgd_theta, sgd_cost_history = stochasticGradientDescent(X, y_column, id=i, alpha=0.01, epochs=10)
		final_cost_history.append(sgd_cost_history)
		print("\033[30mSGD Cost: " + str(sgd_cost_history[-1].flatten()[0]) + "\033[0m")
		gd_theta_df[f'Theta_{i}'] = theta.flatten()
		sgd_theta_df[f'Theta_{i}'] = sgd_theta.flatten()

	plotCostHistory(final_cost_history)

	sgd_theta_df.to_csv("sgd_weights.csv", index=False)
	gd_theta_df.to_csv("gd_weights.csv", index=False)

	print("\033[32mTraining completed. Weights saved to 'sgd_weights.csv' and 'gd_weights.csv'.\033[0m")
