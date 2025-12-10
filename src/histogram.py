import logging
# logging.basicConfig(level=logging.DEBUG)


import matplotlib.pyplot as plt
import pandas as pd
import sys
from describe import describeData, checkInput, readCSV

def plotHistogram(column, column_name):
    print(f"Plotting histogram for column: {column_name}")
    values = {float(v) for v in column if not pd.isna(v)}
    plt.hist(list(values), bins=30, edgecolor='black')
    plt.title(f'Histogram of {column_name}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()
    return values

def main():
    data = readCSV(checkInput())
    describer = describeData(data)
    range_ratio = {}
    # print(describer)
    # all_values = {}
    for column in data.columns:
        if (data[column].dtype == 'float64' or data[column].dtype == 'int64') and data[column].isna().all() == False:
            range_ratio[column] = (describer.at['75%', column] - describer.at['25%', column]) / (describer.at['max', column] - describer.at['min', column])
            plotHistogram(data[column], column)
    print("Range ratios:", range_ratio)
			



if "__main__" == __name__:
    main()