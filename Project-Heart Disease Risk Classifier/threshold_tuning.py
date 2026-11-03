# Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay
from load_data import load_data, clean_data, make_logistic_model
from training import encode, forward_selection, tune_threshold, metrics_at, TARGET_RECALL

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load and clean the data
df = clean_data(load_data())
X = df.drop(columns=['target'])
y = df['target']

# Same split as the other scripts, so the results are comparable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Features: rerun forward selection on the training set (nothing hardcoded, so new data can change them)
features, _, _ = forward_selection(encode(X_train), y_train)
print(f"Forward selection chose {len(features)} features: {features}\n")
model = make_logistic_model(X, features)


# Step 1 + 2: Get honest probabilities for every TRAINING patient, then see how the metrics change
# as the threshold moves. tune_threshold() uses cross_val_predict, so each patient's probability
# comes from a model that never saw that patient. We choose the threshold from these, NOT from
# the test set: picking the threshold that looks best on the test set would be tuning on the
# test set, and its score would no longer be honest.
chosen_threshold, sweep, train_prob = tune_threshold(model, X_train, y_train)

print("=== Threshold sweep on training set (out-of-fold probabilities) ===")
print(f"Sick patients in training set: {y_train.sum()}, healthy: {(y_train == 0).sum()}\n")
print(sweep[sweep['threshold'].isin(np.round(np.arange(0.1, 0.91, 0.1), 2))].set_index('threshold').round(3).to_string())


# Step 3: The chosen threshold.
# Lowering the threshold always raises recall and lowers precision, so tune_threshold() takes the
# HIGHEST threshold that still reaches the target recall (= the fewest false alarms for that recall).
print(f"\nHighest threshold with recall >= {TARGET_RECALL}: {chosen_threshold}")

# For comparison: the threshold that maximizes F1 (balances precision and recall equally)
f1_threshold = sweep.loc[sweep['f1'].idxmax(), 'threshold']
print(f"Threshold with the best F1: {f1_threshold}")


# Step 4: Train on the full training set and check the chosen threshold on the untouched test set
model.fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:, 1]

test_results = pd.DataFrame([
    {'rule': 'Default (0.5)', **metrics_at(y_test, test_prob, 0.5)},
    {'rule': 'Best F1', **metrics_at(y_test, test_prob, f1_threshold)},
    {'rule': f'Recall >= {TARGET_RECALL}', **metrics_at(y_test, test_prob, chosen_threshold)}
]).set_index('rule')

print(f"\n=== Test set ({y_test.sum()} sick, {(y_test == 0).sum()} healthy patients) ===")
print(test_results.round(3).to_string())
print(f"\nTest ROC-AUC: {roc_auc_score(y_test, test_prob):.3f} (the same at every threshold: "
      "it measures how well the model ranks patients, not where we cut)")


# Plot 1: Precision, recall and F1 as the threshold moves
plt.figure(figsize=(9, 5))
plt.plot(sweep['threshold'], sweep['precision'], label='Precision')
plt.plot(sweep['threshold'], sweep['recall'], label='Recall')
plt.plot(sweep['threshold'], sweep['f1'], label='F1', linestyle='--')
plt.axvline(0.5, color='gray', linestyle=':', label='Default threshold 0.5')
plt.axvline(chosen_threshold, color='red', linestyle='--', label=f'Chosen threshold {chosen_threshold}')
plt.axhline(TARGET_RECALL, color='red', alpha=0.3, label=f'Target recall {TARGET_RECALL}')
plt.xlabel('Threshold (predict "disease" when probability >= threshold)')
plt.ylabel('Score')
plt.title('Precision-Recall Trade-off (training set, out-of-fold)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '9_threshold_tradeoff.png'))

# Plot 2: ROC curve on the test set, marking where each threshold sits on it
fpr, tpr, roc_thresholds = roc_curve(y_test, test_prob)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, test_prob):.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random guessing')
for threshold, color in [(0.5, 'gray'), (chosen_threshold, 'red')]:
    point = metrics_at(y_test, test_prob, threshold)
    false_positive_rate = point['false_alarms (FP)'] / (y_test == 0).sum()
    plt.scatter(false_positive_rate, point['recall'], color=color, s=80, zorder=3,
                label=f'Threshold {threshold}')
plt.xlabel('False positive rate (healthy patients flagged)')
plt.ylabel('True positive rate = recall (sick patients caught)')
plt.title('ROC Curve (test set)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '10_roc_curve.png'))

# Plot 3: Confusion matrices at the default and chosen thresholds
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, threshold in zip(axes, [0.5, chosen_threshold]):
    ConfusionMatrixDisplay.from_predictions(
        y_test, (test_prob >= threshold).astype(int),
        display_labels=['No disease', 'Disease'], cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(f'Threshold = {threshold}')
plt.suptitle('Test Set Confusion Matrices (rows = actual, columns = predicted)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '11_confusion_matrices.png'))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
