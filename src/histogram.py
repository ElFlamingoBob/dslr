import logging
# logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
import pandas as pd
from describe import checkInput, readCSV

def plotHistogram(column, column_name):
	print(f"Plotting histogram for column: {column_name}")
	return column.dropna().astype(float).values

def main():
	data = readCSV(checkInput())
	
	fig = plt.figure()
	for i, column in enumerate(data.columns):
		if (data[column].dtype == 'float64' or data[column].dtype == 'int64') and data[column].isna().all() == False and column != 'Index':
			values = plotHistogram(data[column], column)
			ax = fig.add_subplot(4, 4, i + 1)
			ax.hist(list(values), bins=10, edgecolor='black')
			ax.set_title(f'Histogram of {column}')
			ax.set_xlabel('Value')
			ax.set_ylabel('Frequency')
	plt.show()
	




if "__main__" == __name__:
	main()