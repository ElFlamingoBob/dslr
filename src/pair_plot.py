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
	
	g = sns.pairplot(data=data, vars=numeric_columns, hue="Hogwarts House", palette="bright", diag_kind="kde", plot_kws={'alpha':0.7}, corner=True)
	plt.savefig("pair_plot.png", dpi=300)