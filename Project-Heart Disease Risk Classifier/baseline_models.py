# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from load_data import load_data, clean_data, make_preprocessor

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and clean the data
df = clean_data(load_data())

# Features (X) and Target (y)
X = df.drop(columns=['target'])
y = df['target']

# Split data into 80% training and 20% testing.
# stratify=y keeps the same disease/no-disease ratio in both sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training data: {X_train.shape}, Testing data: {X_test.shape}")

# Preprocessing for linear models (one-hot encoding + scaling, see load_data.py).
# Putting this inside a Pipeline means it is fitted on training data only (no leakage).
preprocessor = make_preprocessor(X)


def evaluate(name, y_true, y_pred, y_score):
    """Return the test-set metrics for one model as a dictionary."""
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_score)
    }


results = []


# Model 1: Linear Regression (the "wrong tool" baseline)
# It predicts a continuous number, not a probability, so we threshold it at 0.5.
linear_model = Pipeline([('prep', preprocessor), ('model', LinearRegression())])
linear_model.fit(X_train, y_train)

linear_scores = linear_model.predict(X_test)
linear_pred = (linear_scores >= 0.5).astype(int)
results.append(evaluate('Linear Regression', y_test, linear_pred, linear_scores))

all_linear_scores = linear_model.predict(X)
outside = ((all_linear_scores < 0) | (all_linear_scores > 1)).sum()
print("\n--- Linear Regression ---")
print(f"Prediction range: {all_linear_scores.min():.2f} to {all_linear_scores.max():.2f}")
print(f"Predictions outside [0, 1]: {outside} of {len(X)} -- these can't be probabilities!")


# Model 2: Logistic Regression (the proper linear classifier)
logistic_model = Pipeline([('prep', preprocessor), ('model', LogisticRegression(max_iter=1000))])
logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]
results.append(evaluate('Logistic Regression', y_test, logistic_pred, logistic_prob))

# Coefficients: positive = raises the odds of disease, negative = lowers them
feature_names = logistic_model.named_steps['prep'].get_feature_names_out()
coefficients = pd.Series(logistic_model.named_steps['model'].coef_[0], index=feature_names)
print("\n--- Logistic Regression coefficients (strongest first) ---")
print(coefficients.sort_values(key=abs, ascending=False).round(3))


# Model 3: Shallow Decision Tree (max_depth=3, small enough to read)
# Trees split on thresholds, so they need no scaling or one-hot encoding.
shallow_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
shallow_tree.fit(X_train, y_train)

shallow_pred = shallow_tree.predict(X_test)
shallow_prob = shallow_tree.predict_proba(X_test)[:, 1]
results.append(evaluate('Decision Tree (depth 3)', y_test, shallow_pred, shallow_prob))


# Model 4: Full Decision Tree (no depth limit, grows until every leaf is pure)
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

full_pred = full_tree.predict(X_test)
full_prob = full_tree.predict_proba(X_test)[:, 1]
results.append(evaluate('Decision Tree (full)', y_test, full_pred, full_prob))

# Overfitting check: compare training accuracy with testing accuracy
print("\n--- Overfitting check (train vs test accuracy) ---")
for name, tree in [('Depth 3', shallow_tree), ('Full', full_tree)]:
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc = accuracy_score(y_test, tree.predict(X_test))
    print(f"{name:8s} tree: train = {train_acc:.3f}, test = {test_acc:.3f}, "
          f"depth = {tree.get_depth()}, leaves = {tree.get_n_leaves()}")


# Test-set results
results_df = pd.DataFrame(results).set_index('model')
print("\n--- Test set results (one 80/20 split) ---")
print(results_df.round(3))

# Confusion matrices: rows = actual, columns = predicted, [[TN, FP], [FN, TP]]
print("\n--- Confusion matrices [[TN, FP], [FN, TP]] ---")
for name, pred in [('Linear Regression', linear_pred), ('Logistic Regression', logistic_pred),
                   ('Decision Tree (depth 3)', shallow_pred), ('Decision Tree (full)', full_pred)]:
    cm = confusion_matrix(y_test, pred)
    print(f"{name}: {cm.tolist()}  -> missed {cm[1, 0]} sick patients")


# 5-fold cross-validation: a more reliable estimate than one small test set (60 patients).
# Each patient is used for testing exactly once.
# (Linear Regression is skipped: it isn't a classifier, so sklearn's classification scorers don't apply.)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

cv_results = []
for name, model in [('Logistic Regression', logistic_model),
                    ('Decision Tree (depth 3)', DecisionTreeClassifier(max_depth=3, random_state=42)),
                    ('Decision Tree (full)', DecisionTreeClassifier(random_state=42))]:
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    row = {'model': name}
    for metric in scoring:
        values = scores[f'test_{metric}']
        row[metric] = f"{values.mean():.3f} +/- {values.std():.3f}"
    cv_results.append(row)

print("\n--- 5-fold cross-validation (mean +/- std) ---")
print(pd.DataFrame(cv_results).set_index('model').to_string())


# Plot 1: Linear regression predictions vs actual class
plt.figure(figsize=(7, 4))
jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(y))
plt.scatter(all_linear_scores, y + jitter, alpha=0.5)
plt.axvline(0, color='red', linestyle='--', label='Valid probability range [0, 1]')
plt.axvline(1, color='red', linestyle='--')
plt.axvline(0.5, color='gray', linestyle=':', label='Threshold 0.5')
plt.yticks([0, 1], ['No disease', 'Disease'])
plt.xlabel('Linear regression output')
plt.title('Linear Regression Predictions Escape [0, 1]')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '5_linear_regression_predictions.png'))

# Plot 2: The shallow decision tree, readable like a checklist
plt.figure(figsize=(20, 9))
plot_tree(shallow_tree, feature_names=list(X.columns), class_names=['No disease', 'Disease'],
          filled=True, rounded=True, fontsize=10)
plt.title('Decision Tree (max_depth=3)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '6_decision_tree_depth3.png'))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
