# Import necessary libraries
import os
import sys
import argparse
from datetime import datetime
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from load_data import load_data, clean_data
from training import build_model, metrics_at, TARGET_RECALL, RANDOM_STATE

# Where the trained model is saved (the Flask app loads it from here)
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")

parser = argparse.ArgumentParser(description="Rebuild the heart disease model from the current data.")
parser.add_argument('--force', action='store_true',
                    help="save the new model even if its test recall is lower than the current model's")
args = parser.parse_args()

# Load and clean the data (append new patients to data/heart_cleveland.csv and rerun)
df = clean_data(load_data())
X = df.drop(columns=['target'])
y = df['target']
print(f"Patients: {len(X)} ({y.sum()} with disease, {(y == 0).sum()} without)")


# Step 1: Evaluate the PROCEDURE honestly.
# Run the full procedure (forward selection -> threshold tuning -> logistic regression) on the
# training part only, then score it once on the untouched test part.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print(f"\nStep 1: building the model on the training set ({len(X_train)} patients)...")
evaluation = build_model(X_train, y_train)

test_prob = evaluation['model'].predict_proba(X_test[evaluation['raw_features']])[:, 1]
test_metrics = metrics_at(y_test, test_prob, evaluation['threshold'])
test_metrics['roc_auc'] = roc_auc_score(y_test, test_prob)
print(f"  features ({len(evaluation['features'])}): {evaluation['features']}")
print(f"  threshold: {evaluation['threshold']}")
print(f"  test set ({len(X_test)} patients): recall = {test_metrics['recall']:.3f}, "
      f"precision = {test_metrics['precision']:.3f}, ROC-AUC = {test_metrics['roc_auc']:.3f}, "
      f"missed sick = {test_metrics['missed_sick (FN)']}")


# Step 2: Build the deployed model by running the SAME procedure on ALL patients.
# The features and threshold may differ slightly from step 1; the test score above measures
# the procedure, not one particular list of features.
print(f"\nStep 2: building the final model on all {len(X)} patients...")
final = build_model(X, y)
model = final['model']
print(f"  features ({len(final['features'])}): {final['features']}")
print(f"  raw inputs the app needs ({len(final['raw_features'])}): {final['raw_features']}")
print(f"  threshold: {final['threshold']} (highest with out-of-fold recall >= {TARGET_RECALL})")

# The average patient's features, after encoding + scaling. The app explains a prediction by comparing
# a new patient with this average: contribution = coefficient x (patient - average).
average_features = model[:-1].transform(X[final['raw_features']]).mean()

print("\n  coefficients:")
for name, value in sorted(zip(average_features.index, model[-1].coef_[0]), key=lambda item: -abs(item[1])):
    print(f"    {name:10s} {value:+.3f}")


# Step 3: Compare with the model currently in use
bundle = {
    'model': model,
    'features': final['features'],
    'raw_features': final['raw_features'],
    'threshold': final['threshold'],
    'target_recall': TARGET_RECALL,
    'average_features': average_features.to_dict(),
    'test_metrics': test_metrics,
    'n_patients': len(X),
    'trained_at': datetime.now().isoformat(timespec='seconds')
}

print("\nStep 3: comparing with the current model...")
old = None
if os.path.exists(MODEL_FILE):
    try:
        old = joblib.load(MODEL_FILE)
    except Exception as error:
        # e.g. the file was saved by an older version of this code
        print(f"  the current model.joblib can't be loaded ({error}), so it will be replaced")

if old is None:
    print("  no usable saved model, nothing to compare")
else:
    old_features = old.get('features', [])
    added = [f for f in final['features'] if f not in old_features]
    removed = [f for f in old_features if f not in final['features']]
    print(f"  patients:  {old.get('n_patients')} -> {len(X)}")
    print(f"  features:  {'+ ' + ', '.join(added) if added else 'none added'}; "
          f"{'- ' + ', '.join(removed) if removed else 'none removed'}")
    print(f"  threshold: {old['threshold']} -> {final['threshold']}")
    old_recall = old['test_metrics']['recall']
    print(f"  test recall: {old_recall:.3f} -> {test_metrics['recall']:.3f}")

    # Guard: don't silently replace a model that caught more sick patients.
    # (The two test sets differ when the data grows, so this is a warning sign, not proof.)
    if test_metrics['recall'] < old_recall and not args.force:
        print("\nNOT SAVED: the new model's test recall is lower than the current model's.")
        print("Check the report above, then rerun with --force to replace it anyway.")
        sys.exit(1)


# Step 4: Save everything the app needs in one file
joblib.dump(bundle, MODEL_FILE)
print(f"\nModel saved to: {MODEL_FILE}")
