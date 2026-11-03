# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from load_data import load_data, clean_data, make_logistic_model
from training import (encode, make_cv, backward_elimination, forward_selection,
                      P_VALUE_LIMIT, FORWARD_TOLERANCE)

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and clean the data
df = clean_data(load_data())
X = df.drop(columns=['target'])
y = df['target']

# Same split as baseline_models.py, so the results are comparable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Encode + scale the training data. Feature selection only ever looks at the training set:
# if it saw the test set, the test score would no longer be an honest estimate.
X_train_enc = encode(X_train)
all_features = list(X_train_enc.columns)
print(f"Features after one-hot encoding: {len(all_features)}")
print(all_features)

cv = make_cv()


# Method 1: Backward Elimination (p-values)
# Start with all features, fit a logistic regression, drop the feature with the highest
# p-value, and repeat until every remaining feature is significant (p <= 0.05).
# (The loop itself lives in training.py so other scripts can reuse it.)
print(f"\n=== Backward Elimination (drop highest p-value until all p <= {P_VALUE_LIMIT}) ===")
backward_features, dropped, pvalues = backward_elimination(X_train_enc, y_train)
for feature, p in dropped:
    print(f"Drop {feature:10s} (p = {p:.3f})")

print(f"\nKept {len(backward_features)} features:")
print(pvalues.sort_values().round(4))


# Method 2: Forward Selection (cross-validated ROC-AUC)
# Start with no features. At each step, try adding every remaining feature and keep the one
# that improves the cross-validated ROC-AUC the most. (sklearn's SequentialFeatureSelector
# does the same thing; training.py writes it out so every step is visible.)
print("\n=== Forward Selection (add the feature that improves CV ROC-AUC most) ===")
forward_features, selected, history = forward_selection(X_train_enc, y_train)
for step, (feature, score) in enumerate(zip(selected, history), start=1):
    print(f"Step {step:2d}: add {feature:10s} -> CV ROC-AUC = {score:.4f}")

# Adding every feature always ends at "all features", so forward_selection() chooses where to stop:
# the smallest set whose score is within FORWARD_TOLERANCE of the best (simpler model, nearly the same score).
n_forward = len(forward_features)
print(f"\nBest CV ROC-AUC = {max(history):.4f}. Smallest set within {FORWARD_TOLERANCE} of it: {n_forward} features")
print(forward_features)


# Method 3: LASSO (L1-regularized logistic regression)
# The L1 penalty pushes weak coefficients to exactly zero, so the model selects features itself.
# C controls the penalty strength: small C = strong penalty = fewer features.
# LogisticRegressionCV tries many C values and picks the best by cross-validation.
print("\n=== LASSO (L1 logistic regression, C chosen by CV ROC-AUC) ===")
Cs = np.logspace(-2.5, 1, 30)
lasso = LogisticRegressionCV(Cs=Cs, l1_ratios=[1], solver='liblinear', cv=cv,
                             scoring='roc_auc', max_iter=1000, use_legacy_attributes=False)
lasso.fit(X_train_enc, y_train)

lasso_coefs = pd.Series(lasso.coef_[0], index=all_features)
lasso_features = lasso_coefs[lasso_coefs != 0].index.tolist()
print(f"Best C = {lasso.C_:.4f}")
print(f"\nKept {len(lasso_features)} features, dropped {len(all_features) - len(lasso_features)}:")
print(lasso_coefs.sort_values(key=abs, ascending=False).round(3))


# Compare which features each method kept
feature_sets = {
    'All features': all_features,
    'Backward': backward_features,
    'Forward': forward_features,
    'LASSO': lasso_features
}
comparison = pd.DataFrame({name: ['X' if f in features else '' for f in all_features]
                           for name, features in feature_sets.items() if name != 'All features'},
                          index=all_features)
comparison['votes'] = (comparison == 'X').sum(axis=1)
print("\n=== Features kept by each method (X = kept) ===")
print(comparison.sort_values('votes', ascending=False).to_string())


# Retrain on each feature set and compare.
# Every set uses the same plain logistic regression, so only the features differ.
results = []
for name, features in feature_sets.items():
    model = make_logistic_model(X, features)

    # Held-out test set (never seen during feature selection)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 5-fold CV on all the data. Slightly optimistic for the selected sets, because the
    # selection itself saw the training part of each fold.
    cv_scores = cross_validate(model, X, y, cv=cv, scoring=['recall', 'f1', 'roc_auc'])

    results.append({
        'feature set': name,
        'n_features': len(features),
        'test_accuracy': accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_prob),
        'cv_recall': cv_scores['test_recall'].mean(),
        'cv_f1': cv_scores['test_f1'].mean(),
        'cv_roc_auc': cv_scores['test_roc_auc'].mean()
    })

print("\n=== Logistic Regression on each feature set ===")
print(pd.DataFrame(results).set_index('feature set').round(3).to_string())


# Plot 1: Forward selection curve
plt.figure(figsize=(9, 5))
plt.plot(range(1, len(history) + 1), history, marker='o')
plt.axvline(n_forward, color='red', linestyle='--', label=f'Chosen: {n_forward} features')
for i, (feature, score) in enumerate(zip(selected, history)):
    plt.annotate(feature, (i + 1, score), textcoords='offset points', xytext=(0, 8),
                 ha='center', fontsize=8, rotation=45)
plt.xticks(range(1, len(history) + 1))
plt.ylim(min(history) - 0.01, max(history) + 0.03)  # headroom for the labels
plt.xlabel('Number of features')
plt.ylabel('CV ROC-AUC (training set)')
plt.title('Forward Selection: Score as Features Are Added')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '7_forward_selection_curve.png'))

# Plot 2: LASSO path - how each coefficient shrinks to zero as the penalty gets stronger
path = []
for C in Cs:
    model = LogisticRegression(l1_ratio=1, solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train_enc, y_train)
    path.append(model.coef_[0])
path = np.array(path)

plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(np.linspace(0, 1, len(all_features)))  # 18 lines need more than the default 10 colors
for j, feature in enumerate(all_features):
    plt.plot(Cs, path[:, j], label=feature, color=colors[j])
plt.axvline(lasso.C_, color='black', linestyle='--', label=f'Best C = {lasso.C_:.3f}')
plt.xscale('log')
plt.xlabel('C (left = stronger penalty, fewer features)')
plt.ylabel('Coefficient')
plt.title('LASSO Path: Coefficients Shrink to Zero as the Penalty Grows')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '8_lasso_path.png'))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
