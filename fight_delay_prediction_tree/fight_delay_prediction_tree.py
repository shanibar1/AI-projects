import pandas as pd
import numpy as np
from math import log2
from collections import Counter
from scipy.stats import chi2_contingency

# קריאת הנתונים
data = pd.read_csv('flightdelay.csv')

# חישוב אנטרופיה
def entropy(data):
    total = len(data)
    counts = Counter(data)
    return -sum((count / total) * log2(count / total) for count in counts.values() if count)

# חישוב רווח מידע
def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = sum((len(data[data[feature] == value]) / len(data)) * entropy(data[data[feature] == value][target]) for value in values)
    return total_entropy - weighted_entropy

# מבחן χ² לגיזום העץ
def chi_squared_test(data, feature, target):
    contingency_table = pd.crosstab(data[feature], data[target])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return p

# פונקציה לבחירת התכונה הטובה ביותר
def best_feature(data, features, target):
    return max(features, key=lambda feature: information_gain(data, feature, target))

# מחלקת עץ החלטה
class DecisionTree:
    def __init__(self, depth=0, max_depth=10):
        self.depth = depth
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data, target):
        features = data.columns.drop(target)
        self.tree = self._build_tree(data, features, target)

    def _build_tree(self, data, features, target):
        if len(data[target].unique()) == 1:
            return data[target].values[0]

        if features.empty or self.depth >= self.max_depth:
            return data[target].mode()[0]

        best_feat = best_feature(data, features, target)
        if chi_squared_test(data, best_feat, target) > 0.05:  # רף לחיתוך 0.05
            return data[target].mode()[0]

        tree = {best_feat: {}}
        self.depth += 1

        for value in data[best_feat].unique():
            subset = data[data[best_feat] == value]
            subtree = self._build_tree(subset, features.drop(best_feat), target)
            tree[best_feat][value] = subtree

        return tree

    def predict(self, row):
        tree = self.tree
        while isinstance(tree, dict):
            feature = next(iter(tree))
            value = row[feature]
            tree = tree[feature].get(value, data['DEP_DEL15'].mode()[0])
        return tree

# קיפול צולב ובדיקת איכות העץ
def cross_validation(data, target, k=5):
    fold_size = len(data) // k
    errors = []
    for i in range(k):
        validation_data = data[i*fold_size:(i+1)*fold_size]
        training_data = pd.concat([data[:i*fold_size], data[(i+1)*fold_size:]])
        tree = DecisionTree(max_depth=5)
        tree.fit(training_data, target)
        predictions = validation_data.apply(tree.predict, axis=1)
        error = sum(predictions != validation_data[target]) / len(validation_data)
        errors.append(error)
    return np.mean(errors)

# פונקציות נדרשות לפי הדרישות
def build_tree(ratio):
    train_size = int(len(data) * ratio)
    train_data = data.sample(train_size)
    validation_data = data.drop(train_data.index)
    tree = DecisionTree(max_depth=5)
    tree.fit(train_data, 'DEP_DEL15')
    print(tree.tree)
    predictions = validation_data.apply(tree.predict, axis=1)
    error = sum(predictions != validation_data['DEP_DEL15']) / len(validation_data)
    return error

def tree_error(k):
    return cross_validation(data, 'DEP_DEL15', k)

def is_late(row_input):
    tree = DecisionTree(max_depth=5)
    tree.fit(data, 'DEP_DEL15')
    return tree.predict(row_input)

# בדיקת הפונקציות
print("Building tree with 60% data:")
print(build_tree(0.6))

print("Tree error with 5-fold cross-validation:")
print(tree_error(5))

print("Prediction for the first row in the dataset:")
print(is_late(data.iloc[0]))
