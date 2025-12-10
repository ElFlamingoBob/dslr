import logging
# logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
import pandas as pd
import math
from describe import checkInput, readCSV

def main():
	data = readCSV(checkInput())

	numeric_columns = [col for col in data.columns if (data[col].dtype == 'float64' or data[col].dtype == 'int64') and not data[col].isna().all() and col != 'Index']
	if len(numeric_columns) == 0:
		logging.error("No numeric columns found in the dataset to plot histograms.")
		return
	
	cols = min(5, len(numeric_columns))
	rows = math.ceil(len(numeric_columns) / cols)
	max = rows * cols

	fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 9))
	i = 0
	axes = axes.flatten()
	for column in numeric_columns:
		axes[i].hist(data[column].dropna().astype(float).values, bins=50, color='blue', alpha=0.7)
		axes[i].set_title(f'Histogram of {column}')
		axes[i].set_xlabel(column)
		axes[i].set_ylabel('Frequency')
		i += 1
	
	for j in range(i, max):
		axes[j].axis('off')

	plt.tight_layout()
	plt.show()

if "__main__" == __name__:
	main()