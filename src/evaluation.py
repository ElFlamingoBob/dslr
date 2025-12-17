import pandas as pd
import numpy as np
import sys

from describe import readCSV

def checkInputs():
	if len(sys.argv) != 3:
		print("ERROR Usage: python3 src/evaluation.py <csv_file> <csv_file>")
		sys.exit(1)
	predictions_file = sys.argv[1]
	true_values_file = sys.argv[2]
	if not predictions_file.endswith('.csv') or not true_values_file.endswith('.csv'):
		print("ERROR Input file must be a .csv file")
		sys.exit(1)
	return predictions_file, true_values_file


if "__main__" == __name__:

	predictions_file, true_values_file = checkInputs()
	predictions_data = readCSV(predictions_file)
	true_values_data = readCSV(true_values_file)

	if 'Hogwarts House' not in predictions_data.columns or 'Hogwarts House' not in true_values_data.columns:
		print("ERROR Both CSV files must contain a 'Hogwarts House' column")
		sys.exit(1)

	predicted_houses = predictions_data['Hogwarts House'].to_numpy()
	true_houses = true_values_data['Hogwarts House'].to_numpy()

	if len(predicted_houses) != len(true_houses):
		print("ERROR The number of predictions does not match the number of true values")
		sys.exit(1)

	for i in range(len(true_houses)):
		if (true_houses[i] != predicted_houses[i]):
			print(f"\033[34mMismatch at index {i}: predicted {predicted_houses[i]}, true {true_houses[i]}\033[0m")

	correct_predictions = np.sum(predicted_houses == true_houses)
	total_predictions = len(true_houses)
	accuracy = correct_predictions / total_predictions

	print(f"\033[31mAccuracy: {accuracy * 100:.2f}%\033[0m")