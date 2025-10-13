# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
FDU-PRML-2025Fall is a Pattern Recognition and Machine Learning course repository containing exercises and assignments implementing fundamental ML algorithms from scratch using NumPy.

## Repository Structure

### Exercise1 - Linear Regression
Single and multi-variable linear regression with two solver implementations:
- Closed-form solution (LSE - Least Squares Estimation / Normal Equation)
- Gradient Descent (mini-batch SGD)

**Key files:**
- `Exercise1_Linear_Regression.ipynb` - Main notebook with questions
- `linear_regression.py` - Single variable implementation
- `multi_linear_regression.py` - Multi-variable implementation
- `evaluation.py` - RMSE evaluation script
- `data_generate.py` - Dataset generation

**Data directories:**
- `input/` - Single variable data (train.txt, test_X.txt, test_y.txt)
- `input_multi/` - Multi-variable data
- `output/` - Single variable predictions (SGDpredict.npy, LSEpredict.npy)
- `output_multi/` - Multi-variable predictions

### Assignment1 - Three Parts (100% total)

#### Part 1: Classification Metrics (20%)
Location: `Assignment1/classification/`
- `accuracy_error.py` - Accuracy and MSE implementations
- `evaluation_metrics.py` - Binary classification metrics (precision, recall, F1)
- `test.py` - Test script with built-in test cases

#### Part 2: Decision Tree (40%)
Location: `Assignment1/decision_tree/`
- `criterion.py` - Four splitting criteria implementations:
  - Information Gain
  - Information Gain Ratio
  - Gini Index
  - Classification Error
- `decision_tree.py` - Decision tree classifier with hyperparameters (max_depth, min_samples_split, etc.)
- `test_decision_tree.py` - Test script with 2D toy data
- `viz_tree.py` - Visualization utilities

#### Part 3: k-Nearest Neighbors (40%)
Location: `Assignment1/k_nerest_neighbors/`
- `knn_student.py` - Core kNN implementation with three key functions:
  - `pairwise_dist()` - Distance computation (L2 two_loops/no_loops, cosine)
  - `knn_predict()` - Majority voting prediction
  - `select_k_by_validation()` - Hyperparameter selection via grid search
- `data_generate.py` - 2D multi-class dataset generation
- `test_knn.py` - Unit tests for distance consistency, predictions, tie-breaking
- `viz_knn.py` - Generates k-curve and decision boundary plots
- `input_knn/` - Generated dataset
- `output/` - Plots: knn_k_curve.png, knn_boundary_grid.png

## Common Development Commands

### Running Tests
```bash
# Part 1: Classification metrics
python Assignment1/classification/test.py

# Part 2: Decision tree
python Assignment1/decision_tree/test_decision_tree.py

# Part 3: kNN
python Assignment1/k_nerest_neighbors/test_knn.py
```

### Running Implementations
```bash
# Linear regression (change solver="GD" or "LSE" in main())
python Exercise1/linear_regression.py
python Exercise1/multi_linear_regression.py

# Evaluate predictions
python Exercise1/evaluation.py
```

### Generating Data
```bash
# Linear regression data
python Exercise1/data_generate.py

# kNN data
python Assignment1/k_nerest_neighbors/data_generate.py
```

### Visualization
```bash
# kNN visualization (after implementing knn_student.py)
python Assignment1/k_nerest_neighbors/viz_knn.py

# Decision tree visualization
python Assignment1/decision_tree/viz_tree.py
```

## Implementation Notes

### TODO Pattern
All student implementation tasks are marked with:
```python
# ====================== TODO (students) ======================
# ... instructions or hints ...
raise NotImplementedError("...")  # Delete after implementing
# ====================== END TODO ============================
```

Always remove or comment out `raise NotImplementedError()` after completing TODO sections.

### Data Loading Conventions
- **Linear regression**:
  - train.txt: 2 columns (x, y)
  - test_X.txt: 1 column (x)
  - test_y.txt: 1 column (y)
- **Multi-variable**: Same format but X files have multiple feature columns
- **kNN**: Use `load_prepared_dataset()` from data_generate.py for validation

### Prediction Output Format
- Save predictions as `.npy` files using `np.save()`
- Use dtype=np.float64 for consistency
- Reshape to 1D when needed: `.reshape(-1)`

### Evaluation Scripts
- Linear regression: RMSE metric in `evaluation.py`
- Classification: Accuracy, precision, recall, F1 in test scripts
- kNN: Validation accuracy for k selection

## Architecture Patterns

### Linear Regression Class Structure
```python
class LinearRegression:
    def __init__(self, lr, epochs, batch_size, seed)
    def predict(self, x)  # Vectorized
    def fit(self, x, y, solver, lam=0.0, verbose=True)
    def train_gd(self, x, y, verbose=True)  # Mini-batch SGD
    def train_lse(self, x, y, verbose=True)  # Normal equation
```

### Decision Tree Class Structure
```python
class node:  # Tree node representation
class DecisionTreeClassifier:
    # Hyperparameters: criterion, splitter, max_depth, max_features,
    #                  min_samples_split, min_impurity_decrease, random_state
```

### kNN Function-Based API
Three main functions (no class wrapper):
- Distance matrix computation with multiple metrics/modes
- Prediction via majority voting with tie-breaking (return minimum label)
- Grid search over k values returning best_k and accuracy array

## Jupyter Notebooks
When working with notebooks (especially Exercise1_Linear_Regression.ipynb):
- The notebook includes Questions to answer
- Final submission format: PDF or HTML with code outputs and answers
- Use 当编写可能耗时较长的脚本时，请在脚本中加入打印日志和进度条等功能 (add logging and progress bars for long-running scripts)
