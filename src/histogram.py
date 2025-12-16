import logging
# logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
import seaborn as sns
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
		sns.histplot(data=data, x=column, bins=50, hue="Hogwarts House", alpha=0.7, ax=axes[i], legend=False)
		axes[i].set_title(f'Histogram of {column}')
		axes[i].set_xlabel(column)
		axes[i].set_ylabel('Frequency')
		i += 1

	fig.legend(title='Hogwarts House', labels=data['Hogwarts House'].unique()[::-1], loc='lower right')

	for j in range(i, max):
		axes[j].axis('off')

	plt.tight_layout()
	plt.show()

if "__main__" == __name__:
	main()