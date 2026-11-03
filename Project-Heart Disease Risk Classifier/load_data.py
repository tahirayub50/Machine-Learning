# Import necessary libraries
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# UCI Heart Disease dataset (Cleveland clinic, 303 patients)
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Save a local copy so we only download once
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_FILE = os.path.join(DATA_DIR, "heart_cleveland.csv")

# The raw file has no header row, so we name the 14 columns ourselves
COLUMNS = [
    'age',       # Age in years
    'sex',       # 1 = male, 0 = female
    'cp',        # Chest pain type (1 = typical angina, 2 = atypical, 3 = non-anginal, 4 = asymptomatic)
    'trestbps',  # Resting blood pressure (mm Hg)
    'chol',      # Serum cholesterol (mg/dl)
    'fbs',       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    'restecg',   # Resting ECG result (0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy)
    'thalach',   # Maximum heart rate achieved
    'exang',     # Exercise-induced angina (1 = yes, 0 = no)
    'oldpeak',   # ST depression induced by exercise relative to rest
    'slope',     # Slope of peak exercise ST segment (1 = up, 2 = flat, 3 = down)
    'ca',        # Number of major vessels colored by fluoroscopy (0-3)
    'thal',      # Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    'num'        # Diagnosis: 0 = no disease, 1-4 = disease severity
]

# Columns that are categories stored as numbers (they will need one-hot encoding later)
CATEGORICAL_COLUMNS = ['cp', 'restecg', 'slope', 'thal']

# Every code each categorical column can take. Listing them explicitly keeps the one-hot
# columns the same even when a rare code (restecg = 1 has only 4 patients) is missing from a split.
CATEGORY_CODES = {
    'cp': [1, 2, 3, 4],
    'restecg': [0, 1, 2],
    'slope': [1, 2, 3],
    'thal': [3, 6, 7]
}

# The 13 input features (everything except the target)
FEATURE_COLUMNS = [col for col in COLUMNS if col != 'num']


def load_data():
    """Load the heart disease dataset and return it as a DataFrame with a binary 'target' column.

    To train on more patients later, append rows to data/heart_cleveland.csv
    (same 14 columns) and rerun the scripts. The UCI file is only downloaded if that file is missing.
    """

    # Download the raw data the first time, then reuse the local copy
    if not os.path.exists(DATA_FILE):
        print("Downloading dataset from UCI...")
        os.makedirs(DATA_DIR, exist_ok=True)
        raw = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values='?')
        raw.to_csv(DATA_FILE, index=False)

    df = pd.read_csv(DATA_FILE)

    # Convert diagnosis to a binary target: 0 = no disease, 1 = disease (any severity)
    df['target'] = (df['num'] > 0).astype(int)
    df = df.drop(columns=['num'])

    return df


def clean_data(df):
    """Drop rows with missing values and convert whole-number columns back to int."""

    # Only 6 rows (2%) have missing 'ca' or 'thal', so dropping them loses very little.
    # Dropping also avoids leakage: filling with the mode would use info from the whole
    # dataset (including future test rows) before we split.
    df = df.dropna().reset_index(drop=True)

    # Every column except 'oldpeak' holds whole numbers, so store them as int
    int_columns = [col for col in df.columns if col != 'oldpeak']
    df[int_columns] = df[int_columns].astype(int)

    return df


def make_preprocessor(X):
    """Build the preprocessing step for linear models (fit it on training data only)."""

    # - one-hot encode categorical columns (cp type 4 is not "4 times" cp type 1)
    # - scale the other columns so features with big numbers (chol ~250) don't dominate small ones (oldpeak ~1)
    # Works on any subset of the 13 columns, so the final model can take only the columns it needs.
    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
    numeric_columns = [col for col in X.columns if col not in CATEGORICAL_COLUMNS]
    encoder = OneHotEncoder(categories=[CATEGORY_CODES[col] for col in categorical_columns],
                            drop='first', handle_unknown='ignore', sparse_output=False)
    return ColumnTransformer([
        ('categorical', encoder, categorical_columns),
        ('numeric', StandardScaler(), numeric_columns)
    ], verbose_feature_names_out=False)


def select_columns(data, columns):
    """Keep only the given columns. A named function (not a lambda) so the model can be saved with joblib."""
    return data[columns]


def raw_columns_for(features):
    """Map encoded feature names back to the raw input columns they come from (e.g. 'cp_4' -> 'cp')."""
    raw = set()
    for feature in features:
        prefix = feature.rsplit('_', 1)[0]
        raw.add(prefix if prefix in CATEGORICAL_COLUMNS else feature)
    # Keep the dataset's column order
    return [col for col in FEATURE_COLUMNS if col in raw]


def make_logistic_model(X, features):
    """Pipeline: encode + scale, keep only the chosen (encoded) features, then logistic regression."""
    return Pipeline([
        ('prep', make_preprocessor(X).set_output(transform='pandas')),
        ('select', FunctionTransformer(select_columns, kw_args={'columns': list(features)})),
        ('model', LogisticRegression(max_iter=1000))
    ])


if __name__ == "__main__":
    df = load_data()

    # Display the first few rows of the data
    print(df.head())

    # Shape of the dataset
    print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Data types of each column
    print("\nColumn data types:")
    print(df.dtypes)

    # Missing values per column (only show columns that have some)
    missing = df.isnull().sum()
    print("\nMissing values:")
    print(missing[missing > 0])

    # Duplicate rows (important: duplicates can leak between train and test sets)
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    # Class balance of the target
    print("\nTarget distribution (0 = no disease, 1 = disease):")
    print(df['target'].value_counts())
    print(df['target'].value_counts(normalize=True).round(3))

    # Basic statistics for each feature
    print("\nSummary statistics:")
    print(df.describe().round(2))
