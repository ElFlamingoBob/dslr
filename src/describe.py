import pandas as pd
import math
import sys

pd.set_option('display.float_format', '{:.17g}'.format)

def checkInput():
	if len(sys.argv) != 2:
		print("ERROR Usage: python3 src/describe.py <csv_file>")
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
	nan_count = 0
	zero_count = 0
	tmp_unique_values = []
	for value in column:
		if pd.isna(value):
			nan_count += 1
			continue
		if value == 0.0:
			zero_count += 1
		if value not in tmp_unique_values:
			tmp_unique_values.append(value)
		count += 1
	return {"count": count, "nan_count": nan_count, "zero_count": zero_count, "unique_count": len(tmp_unique_values)}

def describeMean(column, count):
	if count == 0:
		return 0.0
	values = [float(v) for v in column if not pd.isna(v)]
	return math.fsum(values) / count

def describeStdDev(column, mean, count):
	mean_diff_sum = 0.0
	if mean == 0.0 or count <= 1:
		return 0.0
	for v in column: 
		if not pd.isna(v):
			mean_diff_sum += (float(v) - mean) ** 2
	variance = mean_diff_sum / ( count - 1)
	return math.sqrt(variance)

def calculatePercentile(tmp, percentile):
	percentile_index = percentile * (len(tmp) - 1)

	if (percentile_index).is_integer():
		percentile_result = tmp.iloc[int(percentile_index)]
	else:
		floor_id = math.floor(percentile_index)
		fract_part = percentile_index - floor_id
		percentile_result = (tmp.iloc[floor_id] * (1 - fract_part)) + (tmp.iloc[floor_id + 1] * fract_part)

	return percentile_result

def DescribeMinMaxPercentiles(column):
	tmp = column.copy().dropna().sort_values().reset_index(drop=True)
	
	min_value = tmp.iloc[0]
	max_value = tmp.iloc[len(tmp) - 1]

	tfive_percentile = calculatePercentile(tmp, 0.25)
	ffive_percentile = calculatePercentile(tmp, 0.50)
	sfive_percentile = calculatePercentile(tmp, 0.75)

	object = {
		"min": min_value,
		"25%": tfive_percentile,
		"50%": ffive_percentile,
		"75%": sfive_percentile,
		"max": max_value
	}
	return object

def describeData(data):
	describer = pd.DataFrame()
	pd.set_option('display.max_columns', None)

	for column in data.columns:
		if (data[column].dtype == 'float64' or data[column].dtype == 'int64') and data[column].isna().all() == False:
			count_info = describeCount(data[column])
			describer.at['count', column] = count_info["count"]
			describer.at['missing %', column] = (count_info["nan_count"] / (count_info["count"] + count_info["nan_count"])) * 100
			describer.at['zero %', column] = (count_info["zero_count"] / (count_info["count"] + count_info["nan_count"])) * 100
			describer.at['unique', column] = count_info["unique_count"]
			describer.at['mean', column] = describeMean(data[column], describer.at['count', column])
			describer.at['std', column] = describeStdDev(data[column], describer.at['mean', column], describer.at['count', column])
			min_max_percentiles = DescribeMinMaxPercentiles(data[column])
			for key, value in min_max_percentiles.items():
				describer.at[key, column] = value
	return describer

def main():
	describer = describeData(readCSV(checkInput()))
	print(describer)

if __name__ == "__main__":
	main()