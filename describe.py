import pandas as pd
import math
import sys

pd.set_option('display.float_format', '{:.17g}'.format)

def checkInput():
	if len(sys.argv) != 2:
		print("ERROR Usage: python describe.py <csv_file>")
		sys.exit(1)
	filename = sys.argv[1]
	if not filename.endswith('.csv'):
		print("ERROR Input file must be a .csv file")
		sys.exit(1)
	return filename

def readCSV(filename):
	try:
		data = pd.read_csv(filename)
		return data
	except Exception as e:
		print(f"ERROR: {e}")
		sys.exit(1)

def describeCount(column):
	count = 0
	for value in column:
		if pd.isna(value):
			continue
		count += 1
	return count

def describeMean(column, count):
	if count == 0:
		return 0.0
	values = [float(v) for v in column if not pd.isna(v)]
	return math.fsum(values) / count

def 

def describeData(data):
	describer = pd.DataFrame()
	pd.set_option('display.max_columns', None)
	# pd.set_option('display.precision', 20)
	# pd.set_option('display.max_rows', None)
	
	

	for column in data.columns:
		if data[column].dtype == 'float64' or data[column].dtype == 'int64':
			describer.at['count', column] = describeCount(data[column])
			describer.at['mean', column] = describeMean(data[column], describer.at['count', column])
	print(describer)




def main():
	describeData(readCSV(checkInput()))


if __name__ == "__main__":
	main()