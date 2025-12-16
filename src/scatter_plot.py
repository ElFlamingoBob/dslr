import logging
# logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
import seaborn as sns
from describe import checkInput, readCSV

if "__main__" == __name__:
	data = readCSV(checkInput())

	numeric_columns = [col for col in data.columns if (data[col].dtype == 'float64' or data[col].dtype == 'int64') and not data[col].isna().all() and col != 'Index']
	if len(numeric_columns) == 0:
		logging.error("No numeric columns found in the dataset to plot histograms.")
		exit()
	
	corr = data[numeric_columns].corr().abs()
	corr = corr.unstack().sort_values(ascending=False).drop_duplicates()

	for (col1, col2), value in corr.items():
		if (value <= 1.0) and (col1 == col2):
			corr = corr.drop((col1, col2))

	fig, axes = plt.subplots(nrows= 1, ncols= 2, figsize=(10, 6))
	axes[0].set_title('Most Correlated Features')
	axes[1].set_title('Least Correlated Features')
	sns.scatterplot(data=data, x=corr.index[0][0], y=corr.index[0][1], hue="Hogwarts House", palette="bright", alpha=0.7, ax=axes[0])
	sns.scatterplot(data=data, x=corr.index[len(corr)- 1][0], y=corr.index[len(corr)- 1][1], hue="Hogwarts House", palette="bright", alpha=0.7, ax=axes[1], legend=False)

	plt.show()