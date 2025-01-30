# Flight Delay Prediction using Decision Trees

## Project Overview

This project utilizes machine learning techniques to predict flight delays, specifically focusing on the prediction of whether a flight will be delayed or not. The project implements a **Decision Tree** classifier that is trained on historical flight data. The goal is to predict flight delays using various features available in the dataset.

The dataset used for training and testing includes columns like **DEP_DEL15**, which indicates whether a flight was delayed (15 minutes or more). The model is trained using decision tree algorithms and evaluated using cross-validation to assess its performance.

## Key Components

1. **Data Preprocessing**:
   - The dataset is read from a CSV file (`flightdelay.csv`), and the relevant columns are extracted for analysis.
   - Features are selected and processed to build the decision tree classifier.

2. **Entropy and Information Gain**:
   - The project calculates **entropy** and **information gain** to measure the usefulness of each feature in splitting the data. This is fundamental to the decision tree algorithm, as it helps determine which feature to split on at each node.

3. **Chi-Squared Test**:
   - A **Chi-Squared test** is used to prune the decision tree by evaluating the independence of each feature with the target variable. Features with a p-value greater than 0.05 are removed, ensuring that only significant features are retained.

4. **Decision Tree Classifier**:
   - A custom `DecisionTree` class is implemented, which builds the decision tree recursively using information gain and pruning based on the chi-squared test. The tree is built up to a specified maximum depth.
   - The classifier predicts whether a flight will be delayed based on the features of the dataset.

5. **Cross-Validation**:
   - The model's performance is evaluated using **k-fold cross-validation**. The dataset is split into `k` subsets, and the model is trained and tested on each subset. The average error is calculated to give a more robust estimate of the model's performance.

6. **Error Calculation**:
   - The error is calculated by comparing the predicted values with the actual values from the dataset. The `tree_error()` function returns the average error rate after performing cross-validation.

7. **Prediction**:
   - The `is_late()` function predicts whether a specific flight (represented by a row of data) will be delayed or not. The model is trained on the entire dataset and used to make predictions for individual instances.

## Files and Functions

### 1. **`flightdelay.csv`**:
   - This is the dataset used for training the decision tree. It contains flight information, including departure delays and other relevant features.

### 2. **`fight_delay_prediction_tree.py`**:
   - **Entropy**: A function that calculates the entropy of a dataset, used in the calculation of information gain.
   - **Information Gain**: Calculates the information gain for a given feature, which is used to decide the best feature to split on in the decision tree.
   - **Chi-Squared Test**: Performs a Chi-Squared test to determine if a feature is statistically significant for the decision tree.
   - **Decision Tree**: A class that implements the decision tree learning algorithm, including tree-building and prediction functionalities.
   - **Cross-Validation**: Evaluates the model by performing k-fold cross-validation and calculating the error rate.
   - **Prediction**: Makes a prediction for whether a flight will be delayed based on a given input row.

### Example of Execution:

```python
# Building the decision tree using 60% of the data
print("Building tree with 60% data:")
print(build_tree(0.6))

# Performing 5-fold cross-validation to assess the model's performance
print("Tree error with 5-fold cross-validation:")
print(tree_error(5))

# Predicting the delay status for the first flight in the dataset
print("Prediction for the first row in the dataset:")
print(is_late(data.iloc[0]))
