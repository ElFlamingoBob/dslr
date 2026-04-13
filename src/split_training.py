import numpy as np

from describe import checkInput, readCSV

if "__main__" == __name__:
	training_data = readCSV(checkInput("split_training"))

	indices = np.arange(len(training_data))
	np.random.shuffle(indices)
	training_data = training_data.iloc[indices].reset_index(drop=True)

	split_id = int(len(training_data) * 0.80)
	training_set = training_data.iloc[:split_id].reset_index(drop=True)
	validation_set = training_data.iloc[split_id:].reset_index(drop=True)

	print(f"Training set size: {len(training_set)}")
	print(f"Validation set size: {len(validation_set)}")

	training_set.to_csv("./datasets/training_set.csv", index=False)
	validation_set.to_csv("./datasets/validation_set.csv", index=False)