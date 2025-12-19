PYTHON = python3
SRC_DIR = src
DATA_DIR = datasets

TRAIN_DATA = $(DATA_DIR)/dataset_train.csv
TEST_DATA = $(DATA_DIR)/dataset_test.csv
SPLIT_TRAIN = $(DATA_DIR)/training_set.csv
SPLIT_VAL = $(DATA_DIR)/validation_set.csv
GD_WEIGHTS = gd_weights.csv
SGD_WEIGHTS = sgd_weights.csv
MBGD_WEIGHTS = mbgd_weights.csv
PREDICTIONS = houses.csv
DESCRIBE_OUT = describe_output.csv

ORANGE=\033[33m
GREEN=\033[32m
NC=\033[0m

.PHONY: all split describe_train train predict clean describe histogram scatter pair

all: split describe_train train predict

split:
	@$(PYTHON) $(SRC_DIR)/split_training.py $(TRAIN_DATA)

describe_train:
	@echo "$(ORANGE)\nGenerating Descriptive Statistics:$(NC)"
	@$(PYTHON) $(SRC_DIR)/describe.py $(SPLIT_TRAIN)

train:
	@echo "$(ORANGE)\nTraining Logistic Regression Models:$(NC)"
	@$(PYTHON) $(SRC_DIR)/logreg_train.py $(SPLIT_TRAIN)

predict:
	@echo "$(ORANGE)\nPredictions using Gradient Descent Weights:$(NC)"
	@$(PYTHON) $(SRC_DIR)/logreg_predict.py $(SPLIT_VAL) $(GD_WEIGHTS)
	@$(PYTHON) $(SRC_DIR)/evaluation.py $(PREDICTIONS) $(SPLIT_VAL)
	@echo "----------------------------------------"
	@echo "$(ORANGE)Predictions using Stochastic Gradient Descent Weights:$(NC)"
	@$(PYTHON) $(SRC_DIR)/logreg_predict.py $(SPLIT_VAL) $(SGD_WEIGHTS)
	@$(PYTHON) $(SRC_DIR)/evaluation.py $(PREDICTIONS) $(SPLIT_VAL)
	@echo "----------------------------------------"
	@echo "$(ORANGE)Predictions using Mini-batch Gradient Descent Weights:$(NC)"
	@$(PYTHON) $(SRC_DIR)/logreg_predict.py $(SPLIT_VAL) $(MBGD_WEIGHTS)
	@$(PYTHON) $(SRC_DIR)/evaluation.py $(PREDICTIONS) $(SPLIT_VAL)

describe:
	@$(PYTHON) $(SRC_DIR)/describe.py $(TRAIN_DATA)

histogram:
	@$(PYTHON) $(SRC_DIR)/histogram.py $(TRAIN_DATA)

scatter:
	@$(PYTHON) $(SRC_DIR)/scatter_plot.py $(TRAIN_DATA)

pair:
	@$(PYTHON) $(SRC_DIR)/pair_plot.py $(TRAIN_DATA)

clean:
	@rm -f $(SPLIT_TRAIN) $(SPLIT_VAL) $(GD_WEIGHTS) $(SGD_WEIGHTS) $(MBGD_WEIGHTS) $(PREDICTIONS)
	@rm -rf $(SRC_DIR)/__pycache__

clean_all: clean
	@rm -f $(DESCRIBE_OUT) pair_plot.png

re: clean all
