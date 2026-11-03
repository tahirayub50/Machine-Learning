# The model-building procedure, as reusable functions.
# feature_selection.py and threshold_tuning.py use these to explain each step;
# train_final_model.py uses them to rebuild the model automatically on every run.
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from load_data import make_preprocessor, make_logistic_model, raw_columns_for

# Decisions made by people, not learned from data
TARGET_RECALL = 0.90       # catch at least 90% of sick patients
FORWARD_TOLERANCE = 0.005  # forward selection stops at the smallest set within this much of the best score
P_VALUE_LIMIT = 0.05       # backward elimination keeps features with p <= this
RANDOM_STATE = 42


def make_cv():
    """The 5-fold stratified cross-validation used everywhere (fixed seed, so runs are repeatable)."""
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def encode(X):
    """One-hot encode + scale X, returning a DataFrame with the encoded feature names."""
    return make_preprocessor(X).set_output(transform='pandas').fit_transform(X)


def backward_elimination(X_enc, y, p_limit=P_VALUE_LIMIT):
    """Drop the feature with the highest p-value until all p <= p_limit.
    Returns (kept features, list of (dropped feature, p-value), final p-values)."""
    features, dropped = list(X_enc.columns), []
    while True:
        logit = sm.Logit(y, sm.add_constant(X_enc[features])).fit(disp=0)
        pvalues = logit.pvalues.drop('const')
        worst = pvalues.idxmax()
        if pvalues[worst] <= p_limit:
            return features, dropped, pvalues
        dropped.append((worst, pvalues[worst]))
        features.remove(worst)


def forward_selection(X_enc, y, tolerance=FORWARD_TOLERANCE):
    """Add the feature that most improves CV ROC-AUC, one at a time, until all are added.
    Returns (chosen features, full order of features added, CV score after each step)."""
    cv = make_cv()

    def cv_auc(features):
        model = LogisticRegression(max_iter=1000)
        return cross_val_score(model, X_enc[features], y, cv=cv, scoring='roc_auc').mean()

    order, remaining, history = [], list(X_enc.columns), []
    while remaining:
        scores = {feature: cv_auc(order + [feature]) for feature in remaining}
        best = max(scores, key=scores.get)
        order.append(best)
        remaining.remove(best)
        history.append(scores[best])

    # Adding every feature always ends at "all features", so choose where to stop:
    # the smallest set whose score is within `tolerance` of the best (simpler model, nearly the same score).
    best_score = max(history)
    n_chosen = next(i + 1 for i, score in enumerate(history) if score >= best_score - tolerance)
    return order[:n_chosen], order, history


def metrics_at(y_true, y_prob, threshold):
    """Classify as 'disease' when probability >= threshold, and return the metrics."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'missed_sick (FN)': int(fn),
        'false_alarms (FP)': int(fp)
    }


def tune_threshold(model, X, y, target_recall=TARGET_RECALL):
    """Pick the highest threshold whose out-of-fold recall still reaches target_recall.
    Returns (threshold, sweep table, out-of-fold probabilities)."""

    # Each patient's probability comes from a model that never saw that patient
    prob = cross_val_predict(model, X, y, cv=make_cv(), method='predict_proba')[:, 1]

    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    sweep = pd.DataFrame([metrics_at(y, prob, t) for t in thresholds])

    # Lowering the threshold always raises recall and lowers precision, so the HIGHEST
    # threshold that still reaches the target gives the fewest false alarms for that recall
    reaching_target = sweep[sweep['recall'] >= target_recall]
    threshold = float(reaching_target['threshold'].max()) if len(reaching_target) else float(thresholds[0])
    return threshold, sweep, prob


def build_model(X, y):
    """The full procedure: forward selection -> logistic regression on the chosen features -> threshold.
    Everything is learned from (X, y) only. Returns a dict with the fitted model and its choices."""
    features, _, _ = forward_selection(encode(X), y)

    # The model only needs the raw columns its features come from
    raw_features = raw_columns_for(features)
    model = make_logistic_model(X[raw_features], features)

    threshold, _, _ = tune_threshold(model, X[raw_features], y)
    model.fit(X[raw_features], y)

    return {
        'model': model,
        'features': features,
        'raw_features': raw_features,
        'threshold': threshold
    }
