# Mathematics in DSLR

A precise reference for every mathematical operation used in this repository.  
Each section maps directly to the source file(s) where it is implemented.

---

## Table of Contents

1. [Descriptive Statistics](#1-descriptive-statistics)
   - 1.1 [Count & Missing Percentage](#11-count--missing-percentage)
   - 1.2 [Zero Percentage](#12-zero-percentage)
   - 1.3 [Arithmetic Mean](#13-arithmetic-mean)
   - 1.4 [Sample Standard Deviation](#14-sample-standard-deviation)
   - 1.5 [Percentiles — Linear Interpolation](#15-percentiles--linear-interpolation)
   - 1.6 [Minimum & Maximum](#16-minimum--maximum)
2. [Feature Normalisation (Z-Score Standardisation)](#2-feature-normalisation-z-score-standardisation)
3. [Sigmoid (Logistic) Function](#3-sigmoid-logistic-function)
4. [Hypothesis Function](#4-hypothesis-function)
5. [Binary Cross-Entropy Cost Function](#5-binary-cross-entropy-cost-function)
6. [Gradient Descent (GD)](#6-gradient-descent-gd)
   - 6.1 [Gradient Computation](#61-gradient-computation)
   - 6.2 [Weight Update Rule](#62-weight-update-rule)
7. [Stochastic Gradient Descent (SGD)](#7-stochastic-gradient-descent-sgd)
   - 7.1 [Per-Sample Gradient](#71-per-sample-gradient)
   - 7.2 [Learning-Rate Decay](#72-learning-rate-decay)
8. [Mini-Batch Gradient Descent (MBGD)](#8-mini-batch-gradient-descent-mbgd)
9. [One-vs-All (OvA) Multi-Class Classification](#9-one-vs-all-ova-multi-class-classification)
10. [Pearson Correlation Coefficient](#10-pearson-correlation-coefficient)
11. [Accuracy Metric](#11-accuracy-metric)
12. [Train / Validation Split](#12-train--validation-split)

---

## 1. Descriptive Statistics

**Source:** `src/describe.py`

---

### 1.1 Count & Missing Percentage

For a column of *N* total rows containing *n* non-null (valid) values and *k* null (NaN) values, where *N = n + k*:

$$\text{count} = n$$

$$\text{missing \%} = \frac{k}{N} \times 100 = \frac{k}{n + k} \times 100$$

**Code:** `describeCount()` iterates through the column, incrementing `nan_count` for every `pd.isna(v)` entry and `count` for every valid entry.  The missing percentage is then computed as `(nan_count / (count + nan_count)) * 100`.

---

### 1.2 Zero Percentage

$$\text{zero \%} = \frac{z}{N} \times 100$$

where *z* is the number of values equal to exactly `0.0`, and *N* is the total number of rows (including nulls).

**Code:** `zero_count` is accumulated inside `describeCount()` and the final ratio uses the same denominator `(count + nan_count)`.

---

### 1.3 Arithmetic Mean

Given *n* valid (non-null) values $x_1, x_2, \ldots, x_n$:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Code:** `describeMean()` uses Python's `math.fsum()` for compensated (numerically stable) summation to reduce floating-point rounding errors, then divides by `count`.

---

### 1.4 Sample Standard Deviation

The **sample** (Bessel-corrected) standard deviation uses *n − 1* in the denominator:

$$s = \sqrt{\frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n - 1}}$$

The intermediate quantity $\sum_{i=1}^{n}(x_i - \bar{x})^2$ is the **sum of squared deviations** from the mean.  
Dividing by *n − 1* instead of *n* gives an **unbiased estimate** of the population variance.  
Taking the square root converts variance (units²) back to the original units.

**Code:** `describeStdDev()` accumulates `mean_diff_sum += (v - mean) ** 2`, computes `variance = mean_diff_sum / (count - 1)`, and returns `math.sqrt(variance)`.  Returns `0.0` when `count <= 1` (undefined for a single point).

---

### 1.5 Percentiles — Linear Interpolation

To find the *p*-th percentile (0 ≤ p ≤ 1) of a sorted array of *n* values $v_0, v_1, \ldots, v_{n-1}$:

1. Compute the **real-valued index**:

$$i_p = p \times (n - 1)$$

2. If $i_p$ is an integer, the percentile is the value at that index:

$$P_p = v_{i_p}$$

3. Otherwise, let $f = \lfloor i_p \rfloor$ (the floor index) and $\delta = i_p - f$ (the fractional part). **Linear interpolation** between the two surrounding values:

$$P_p = v_f \times (1 - \delta) + v_{f+1} \times \delta$$

This is the same method used by NumPy's `np.percentile` with `interpolation='linear'` (the default).

The three quartiles computed are:

| Statistic | *p* value |
|-----------|-----------|
| 25% (Q1)  | 0.25      |
| 50% (Q2, median) | 0.50 |
| 75% (Q3)  | 0.75      |

**Code:** `calculatePercentile(tmp, percentile)` where `tmp` is the sorted, NaN-dropped column.

---

### 1.6 Minimum & Maximum

$$\min = v_0 \quad \text{(first element of the sorted column)}$$
$$\max = v_{n-1} \quad \text{(last element of the sorted column)}$$

**Code:** After calling `sort_values()`, `tmp.iloc[0]` and `tmp.iloc[len(tmp) - 1]` are read directly.

---

## 2. Feature Normalisation (Z-Score Standardisation)

**Source:** `src/logreg_train.py` — `normalizeFeatures()`

Each feature column $j$ is standardised independently using the **mean** $\bar{x}_j$ and **sample standard deviation** $s_j$ computed by `describe.py` and stored in `describe_output.csv`:

$$x'_{ij} = \frac{x_{ij} - \bar{x}_j}{s_j}$$

After normalisation every feature has approximately **zero mean** and **unit variance**, which prevents features with larger numeric ranges from dominating the gradient and makes gradient descent converge faster and more reliably.

In matrix form, where **X** is the (*m* × *n*) feature matrix:

$$\mathbf{X'} = \frac{\mathbf{X} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

where $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ are row vectors of per-feature means and standard deviations respectively, and the subtraction/division are element-wise (broadcasting).

**Code:** `X_norm = (X - means) / std_devs` — a single vectorised NumPy expression.

---

## 3. Sigmoid (Logistic) Function

**Source:** `src/logreg_train.py` — `sigmoid(z)`

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

where *e* ≈ 2.71828 is Euler's number.

**Properties:**

| Property | Value |
|----------|-------|
| Domain | $(-\infty, +\infty)$ |
| Range | $(0, 1)$ — strictly open, never exactly 0 or 1 |
| $\sigma(0)$ | 0.5 |
| As $z \to +\infty$ | $\sigma(z) \to 1$ |
| As $z \to -\infty$ | $\sigma(z) \to 0$ |
| Derivative | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |

The output is interpreted as a **probability**: $P(y=1 \mid \mathbf{x}; \boldsymbol{\theta}) = \sigma(z)$.

**Code:** `1 / (1 + np.exp(-z))` applied element-wise via NumPy broadcasting.

---

## 4. Hypothesis Function

**Source:** `src/logreg_train.py` — `hypothesis(X, theta)`

For a single sample $\mathbf{x} \in \mathbb{R}^{n+1}$ (with a leading bias term of 1) and weight vector $\boldsymbol{\theta} \in \mathbb{R}^{n+1}$:

$$h_{\boldsymbol{\theta}}(\mathbf{x}) = \sigma\!\left(\boldsymbol{\theta}^{\top} \mathbf{x}\right) = \sigma\!\left(\sum_{j=0}^{n} \theta_j x_j\right)$$

For the full training matrix $\mathbf{X} \in \mathbb{R}^{m \times (n+1)}$ (each row is one sample):

$$\mathbf{h} = \sigma(\mathbf{X}\boldsymbol{\theta}) \in \mathbb{R}^{m \times 1}$$

The dot product $\mathbf{X}\boldsymbol{\theta}$ computes the **linear combination** (also called the *logit* or *pre-activation*) for every sample at once.

**Code:** `sigmoid(np.dot(X, theta))`.

---

## 5. Binary Cross-Entropy Cost Function

**Source:** `src/logreg_train.py` — computed inline inside each training function

For *m* samples with true labels $\mathbf{y} \in \{0, 1\}^m$ and predicted probabilities $\mathbf{h} = h_{\boldsymbol{\theta}}(\mathbf{X})$:

$$J(\boldsymbol{\theta}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log\!\left(h^{(i)}\right) + \left(1 - y^{(i)}\right) \log\!\left(1 - h^{(i)}\right) \right]$$

In matrix notation:

$$J(\boldsymbol{\theta}) = -\frac{1}{m} \left[ \mathbf{y}^{\top} \log(\mathbf{h}) + (\mathbf{1} - \mathbf{y})^{\top} \log(\mathbf{1} - \mathbf{h}) \right]$$

**Intuition:**

- When $y^{(i)} = 1$: the cost is $-\log(h^{(i)})$, which is low when $h^{(i)} \approx 1$ and very high as $h^{(i)} \to 0$.
- When $y^{(i)} = 0$: the cost is $-\log(1 - h^{(i)})$, which is low when $h^{(i)} \approx 0$ and very high as $h^{(i)} \to 1$.

This cost function is derived from the **negative log-likelihood** of a Bernoulli distribution and is **convex** in $\boldsymbol{\theta}$, guaranteeing a global minimum.

**Code (GD):** `(-1 / m) * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))`  
**Code (SGD):** `(-1) * (yi.T * np.log(h) + (1 - yi).T * np.log(1 - h))` — same formula with $m=1$

---

## 6. Gradient Descent (GD)

**Source:** `src/logreg_train.py` — `gradientDescent()`

Parameters: learning rate `alpha = 0.01`, `num_iterations = 4000`.

### 6.1 Gradient Computation

The gradient of $J(\boldsymbol{\theta})$ with respect to $\boldsymbol{\theta}$ is:

$$\nabla_{\boldsymbol{\theta}} J = \frac{1}{m} \mathbf{X}^{\top} (\mathbf{h} - \mathbf{y})$$

Derivation sketch:

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left(h^{(i)} - y^{(i)}\right) x_j^{(i)}$$

Stacking all partial derivatives into a vector and writing in matrix form gives the expression above. The fact that $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ causes the chain-rule terms to cancel cleanly, leaving the elegantly simple residual form $(\mathbf{h} - \mathbf{y})$.

**Code:** `gradient = np.dot(X.T, (h - y)) / m`

### 6.2 Weight Update Rule

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha \cdot \nabla_{\boldsymbol{\theta}} J$$

Each iteration moves $\boldsymbol{\theta}$ by a step of size $\alpha$ (the **learning rate**) in the direction of steepest descent.

**Code:** `theta = theta - (alpha * gradient)`

---

## 7. Stochastic Gradient Descent (SGD)

**Source:** `src/logreg_train.py` — `stochasticGradientDescent()`

Parameters: `alpha = 0.01`, `epochs = 10`, `decay_rate = 0.001`.

### 7.1 Per-Sample Gradient

Instead of summing over all *m* samples, a single randomly selected sample $(\mathbf{x}^{(i)}, y^{(i)})$ is used per update:

$$\nabla_{\boldsymbol{\theta}} J^{(i)} = \mathbf{x}^{(i)\top} \left(h^{(i)} - y^{(i)}\right)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_t \cdot \nabla_{\boldsymbol{\theta}} J^{(i)}$$

The data is **shuffled** at the start of each epoch using a random permutation of indices, ensuring that over one epoch every sample is visited exactly once.

**Code:**
```python
indices = np.random.permutation(m)
gradient = np.dot(xi.T, (h - yi))   # xi shape (1, n+1), h and yi shape (1,1)
theta = theta - (current_alpha * gradient)
```

### 7.2 Learning-Rate Decay

To improve convergence as training progresses, the learning rate is reduced each epoch using **inverse-time decay**:

$$\alpha_t = \frac{\alpha_0}{1 + \lambda \cdot t}$$

where:
- $\alpha_0 = 0.01$ is the initial learning rate
- $\lambda = 0.001$ is the decay rate
- $t$ is the current epoch number (0-indexed)

This schedule ensures the step size decreases monotonically, reducing oscillations near the minimum in later epochs.

**Code:** `current_alpha = alpha / (1 + decay_rate * epoch)`

---

## 8. Mini-Batch Gradient Descent (MBGD)

**Source:** `src/logreg_train.py` — `minibatchGradienDescent()`

Parameters: `alpha = 0.01`, `epochs = 10`, `batch_size = 16`, `decay_rate = 0.001`.

MBGD is a compromise between full-batch GD and SGD. The training set is divided into **mini-batches** of size $B = 16$ samples.  For each batch $\mathcal{B}_k$ of size $m_k$ (the last batch may be smaller):

$$\nabla_{\boldsymbol{\theta}} J_{\mathcal{B}_k} = \frac{1}{m_k} \mathbf{X}_{\mathcal{B}_k}^{\top} \left(\mathbf{h}_{\mathcal{B}_k} - \mathbf{y}_{\mathcal{B}_k}\right)$$

$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha_t \cdot \nabla_{\boldsymbol{\theta}} J_{\mathcal{B}_k}$$

The same **inverse-time learning-rate decay** as SGD applies (same formula, same hyperparameters).  
Data is also **shuffled** each epoch.

**Code:**
```python
gradient = np.dot(X_batch.T, (h - y_batch)) / m_batch
theta = theta - (current_alpha * gradient)
```

**Comparison of optimisers:**

| Property | GD | SGD | MBGD |
|---|---|---|---|
| Gradient computed over | All *m* samples | 1 sample | *B* = 16 samples |
| Update frequency | Once per epoch | *m* times per epoch | *m/B* times per epoch |
| Noise | Low (stable) | High | Moderate |
| Learning-rate decay | No | Yes | Yes |
| Data shuffle per epoch | No | Yes | Yes |

---

## 9. One-vs-All (OvA) Multi-Class Classification

**Source:** `src/logreg_train.py`, `src/logreg_predict.py`

With $K = 4$ classes (Gryffindor, Hufflepuff, Ravenclaw, Slytherin), four **independent binary classifiers** are trained.  
Classifier $k$ is trained with a modified label vector:

$$y_k^{(i)} = \begin{cases} 1 & \text{if sample } i \text{ belongs to class } k \\ 0 & \text{otherwise} \end{cases}$$

This is the `yMapping()` function: one class is set to 1, all others to 0.

After training, prediction for a new sample $\mathbf{x}$ is:

$$\hat{y} = \arg\max_{k \in \{0,1,2,3\}} h_{\boldsymbol{\theta}_k}(\mathbf{x})$$

i.e. the class whose trained classifier outputs the **highest probability** wins.

**Code:** `predicted_classes = np.argmax(predictions, axis=1)` where `predictions` is an $(m \times 4)$ matrix of per-house probabilities.

---

## 10. Pearson Correlation Coefficient

**Source:** `src/scatter_plot.py`

Used to rank feature pairs by linear correlation in order to display the most and least correlated pair.

For two features $X$ and $Y$ over *n* observations:

$$r_{XY} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

The coefficient satisfies $-1 \leq r_{XY} \leq 1$:

| Value | Interpretation |
|-------|----------------|
| $r = 1$ | Perfect positive linear relationship |
| $r = -1$ | Perfect negative linear relationship |
| $r = 0$ | No linear relationship |

The code calls `.abs()` on the correlation matrix, so the ranking is by the **magnitude** of correlation (regardless of direction). Self-correlations ($r_{XX} = 1$) are removed before finding the most and least correlated pair.

**Code:** `corr = data[numeric_columns].corr().abs()` — Pandas internally computes Pearson's *r*.

---

## 11. Accuracy Metric

**Source:** `src/evaluation.py`

$$\text{Accuracy} = \frac{\text{number of correct predictions}}{\text{total number of predictions}} = \frac{\sum_{i=1}^{N} \mathbf{1}[\hat{y}^{(i)} = y^{(i)}]}{N}$$

where $\mathbf{1}[\cdot]$ is the indicator function (1 if the condition is true, 0 otherwise).

**Code:** `accuracy = np.sum(predicted_houses == true_houses) / len(true_houses)`

The result is multiplied by 100 and printed as a percentage, e.g. `98.50%`.

---

## 12. Train / Validation Split

**Source:** `src/split_training.py`

The full labelled dataset of *N* rows is partitioned into:

$$N_{\text{train}} = \lfloor 0.85 \times N \rfloor \quad \text{(first 85 % of shuffled rows)}$$
$$N_{\text{val}} = N - N_{\text{train}} \quad \text{(remaining 15 % of shuffled rows)}$$

Before splitting, the dataset is **randomly shuffled** using a random permutation of row indices.  This ensures that the split is unbiased and that neither subset contains systematic ordering artefacts from the original file.

**Code:**
```python
indices = np.arange(len(training_data))
np.random.shuffle(indices)
split_id = int(len(training_data) * 0.85)
training_set   = training_data.iloc[:split_id]
validation_set = training_data.iloc[split_id:]
```

---

## Summary Table

| Concept | Formula (compact) | Source file |
|---|---|---|
| Count | $n = \sum_i \mathbf{1}[x_i \neq \text{NaN}]$ | `describe.py` |
| Missing % | $(k / N) \times 100$ | `describe.py` |
| Zero % | $(z / N) \times 100$ | `describe.py` |
| Mean | $\bar{x} = \frac{1}{n}\sum x_i$ | `describe.py` |
| Sample std dev | $s = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}$ | `describe.py` |
| Percentile | $P_p = v_f(1-\delta) + v_{f+1}\delta$ | `describe.py` |
| Z-score normalisation | $x' = (x - \bar{x}) / s$ | `logreg_train.py` |
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | `logreg_train.py` |
| Hypothesis | $h = \sigma(\mathbf{X}\boldsymbol{\theta})$ | `logreg_train.py` |
| Cost (cross-entropy) | $J = -\frac{1}{m}[\mathbf{y}^{\top}\log\mathbf{h} + (1-\mathbf{y})^{\top}\log(1-\mathbf{h})]$ | `logreg_train.py` |
| GD gradient | $\nabla J = \frac{1}{m}\mathbf{X}^{\top}(\mathbf{h}-\mathbf{y})$ | `logreg_train.py` |
| Weight update | $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\nabla J$ | `logreg_train.py` |
| LR decay | $\alpha_t = \alpha_0 / (1 + \lambda t)$ | `logreg_train.py` |
| OvA prediction | $\hat{y} = \arg\max_k h_k(\mathbf{x})$ | `logreg_predict.py` |
| Pearson correlation | $r = \frac{\sum(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2\sum(y_i-\bar{y})^2}}$ | `scatter_plot.py` |
| Accuracy | $\sum_i\mathbf{1}[\hat{y}^{(i)}=y^{(i)}] / N$ | `evaluation.py` |
| Train/val split | $N_{\text{train}} = \lfloor 0.85N \rfloor$ | `split_training.py` |
