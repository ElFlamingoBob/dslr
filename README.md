<div align="center">

# 🧙 DSLR — Data Science x Logistic Regression

**A Harry Potter–themed multi-class logistic regression classifier built from scratch.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-used-013243?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-used-150458?logo=pandas)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-used-11557c)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-used-4c72b0)](https://seaborn.pydata.org/)

</div>

---

## 📖 Overview

**DSLR** is a machine learning project from the 42 curriculum. The goal is to predict which of the four Hogwarts houses a student belongs to — **Gryffindor**, **Hufflepuff**, **Ravenclaw**, or **Slytherin** — by training a one-vs-all logistic regression classifier **entirely from scratch**, without using any high-level ML library for the model itself.

Three optimization strategies are implemented and compared:

| Method | Description |
|---|---|
| **Gradient Descent (GD)** | Classic full-batch update over all training samples |
| **Stochastic Gradient Descent (SGD)** | Per-sample updates with learning-rate decay |
| **Mini-batch Gradient Descent (MBGD)** | Updates over small randomized batches with learning-rate decay |

---

## 🗂️ Project Structure

```
dslr/
├── datasets/
│   ├── dataset_train.csv      # Full labelled training data
│   └── dataset_test.csv       # Unlabelled test data
├── src/
│   ├── describe.py            # Custom statistical describe (no pandas.describe)
│   ├── histogram.py           # Distribution histograms per house
│   ├── scatter_plot.py        # Most / least correlated feature scatter plots
│   ├── pair_plot.py           # Full pair-plot saved as pair_plot.png
│   ├── split_training.py      # 95/5 train–validation split
│   ├── logreg_train.py        # Logistic regression training (GD, SGD, MBGD)
│   ├── logreg_predict.py      # House prediction from saved weights
│   └── evaluation.py         # Accuracy evaluation against ground truth
├── Makefile                   # Convenience build targets
└── README.md
```

---

## ⚙️ Requirements

| Package | Purpose |
|---|---|
| `numpy` | Matrix operations & sigmoid |
| `pandas` | CSV I/O & data manipulation |
| `matplotlib` | Plotting |
| `seaborn` | Advanced visualization |

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn
```

---

## 🚀 Quick Start

### Full Pipeline (split → describe → train → predict)

```bash
make all
```

### Step by Step

```bash
# 1. Split dataset_train.csv into 95 % training / 5 % validation sets
make split

# 2. Print descriptive statistics on the training split
make describe_train

# 3. Train all three models — saves gd_weights.csv, sgd_weights.csv, mbgd_weights.csv
make train

# 4. Run predictions and print accuracy for each method
make predict
```

---

## 📊 Data Visualization

Explore the dataset before training with three visualization tools.

### Histograms — score distribution per house
```bash
make histogram
# or
python3 src/histogram.py datasets/dataset_train.csv
```

Displays a grid of histograms for every numeric course feature, coloured by Hogwarts house.

### Scatter Plot — feature correlation
```bash
make scatter
# or
python3 src/scatter_plot.py datasets/dataset_train.csv
```

Shows the **most correlated** and **least correlated** pair of features side by side.

### Pair Plot — full feature matrix
```bash
make pair
# or
python3 src/pair_plot.py datasets/dataset_train.csv
```

Renders a lower-triangular pair-plot and saves it to `pair_plot.png`.

![Pair Plot](pair_plot.png)

---

## 📐 Descriptive Statistics

The `describe.py` script reimplements `pandas.DataFrame.describe()` from scratch, computing:

| Statistic | Description |
|---|---|
| `count` | Non-null value count |
| `missing %` | Percentage of missing values |
| `zero %` | Percentage of zero values |
| `unique` | Number of unique values |
| `mean` | Arithmetic mean |
| `std` | Sample standard deviation |
| `min` / `max` | Range bounds |
| `25%` / `50%` / `75%` | Percentiles via linear interpolation |

```bash
make describe
# or
python3 src/describe.py datasets/dataset_train.csv
```

Output is also saved to `describe_output.csv` and consumed during feature normalisation.

---

## 🧠 Model Details

### One-vs-All Logistic Regression

Four binary classifiers are trained — one per house. Each classifier learns to distinguish its house from all others using:

- **Sigmoid function**: σ(z) = 1 / (1 + e⁻ᶻ)
- **Binary cross-entropy loss**
- **Feature normalisation** using pre-computed mean and standard deviation

Prediction selects the house whose classifier produces the highest probability.

### Gradient Descent Variants

All three optimisers share the same hypothesis and cost function, differing only in how weight updates are computed:

```
θ := θ − α · ∇J(θ)
```

- **GD** — gradient computed over the full training set
- **SGD** — gradient computed one sample at a time; shuffles data each epoch; applies learning-rate decay
- **MBGD** — gradient computed over batches of 16; shuffles data each epoch; applies learning-rate decay

After training, weights are saved to `gd_weights.csv`, `sgd_weights.csv`, and `mbgd_weights.csv`.

---

## 🔮 Prediction

```bash
python3 src/logreg_predict.py <dataset.csv> <weights.csv>
```

**Examples:**
```bash
python3 src/logreg_predict.py datasets/validation_set.csv gd_weights.csv
python3 src/logreg_predict.py datasets/validation_set.csv sgd_weights.csv
python3 src/logreg_predict.py datasets/validation_set.csv mbgd_weights.csv
```

Predictions are written to `houses.csv`.

---

## ✅ Evaluation

```bash
python3 src/evaluation.py houses.csv datasets/validation_set.csv
```

Prints per-row mismatches (if any) and overall accuracy:

```
Accuracy: 98.50%
```

---

## 🛠️ Makefile Reference

| Target | Description |
|---|---|
| `make all` | `split` → `describe_train` → `train` → `predict` |
| `make split` | Split `dataset_train.csv` into train / validation sets |
| `make describe_train` | Describe the training split |
| `make describe` | Describe the full `dataset_train.csv` |
| `make train` | Train GD, SGD, and MBGD models |
| `make predict` | Run predictions and evaluate all three models |
| `make histogram` | Show histograms |
| `make scatter` | Show scatter plots |
| `make pair` | Generate and save pair plot |
| `make clean` | Remove split files, weights, and predictions |
| `make clean_all` | `clean` + remove describe output and pair plot |
| `make re` | `clean` then `all` |

---

## 📄 License

This project is part of the [42 School](https://42.fr) curriculum and is intended for educational purposes.