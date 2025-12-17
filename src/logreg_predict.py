import logging
# logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import sys

from describe import readCSV
from logreg_train import normalizeFeatures, hypothesis

# from sklearn.metrics import accuracy_score

def checkInputs():
	if len(sys.argv) != 3:
		print("ERROR Usage: python3 src/logreg_predict.py <csv_file> <weights_file>")
		sys.exit(1)
	test_file = sys.argv[1]
	weights_file = sys.argv[2]
	if not test_file.endswith('.csv') or not weights_file.endswith('.csv'):
		print("ERROR Input file must be a .csv file")
		sys.exit(1)
	return test_file, weights_file



if "__main__" == __name__:
	test_file, weights_file = checkInputs()
	data = readCSV(test_file)
	weights = readCSV(weights_file)

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

	predictions = []

	for i in range(4):
		theta = weights[f'Theta_{i}'].to_numpy().reshape(-1, 1)
		h = hypothesis(X, theta)
		predictions.append(h.flatten())

	predictions = np.array(predictions).T
	predicted_classes = np.argmax(predictions, axis=1)

	mapping = {0: 'Gryffindor', 1: 'Hufflepuff', 2: 'Ravenclaw', 3: 'Slytherin'}
	predicted_houses = [mapping[i] for i in predicted_classes]
	output_df = pd.DataFrame({'Index': data['Index'], 'Hogwarts House': predicted_houses})
	output_df.to_csv("houses.csv", index=False)


	
