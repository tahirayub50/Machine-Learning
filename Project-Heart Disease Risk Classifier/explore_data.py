# Import necessary libraries
import os
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load_data, clean_data

# Folder to save the plots
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Continuous features (real measurements) vs categorical features (groups stored as numbers)
NUMERIC_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORY_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Load and clean the data
df = clean_data(load_data())
print(f"Rows after cleaning: {df.shape[0]} (dropped {303 - df.shape[0]} rows with missing values)")


# 1. Class balance
print("\nTarget distribution (0 = no disease, 1 = disease):")
print(df['target'].value_counts())

plt.figure(figsize=(5, 4))
sns.countplot(x='target', data=df)
plt.xticks([0, 1], ['No disease', 'Disease'])
plt.title('Class Balance')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '1_class_balance.png'))


# 2. Numeric features: how do their distributions differ between the two classes?
print("\nAverage of numeric features by class:")
print(df.groupby('target')[NUMERIC_FEATURES].mean().round(2))

fig, axes = plt.subplots(1, len(NUMERIC_FEATURES), figsize=(20, 4))
for ax, col in zip(axes, NUMERIC_FEATURES):
    sns.histplot(data=df, x=col, hue='target', kde=True, ax=ax)
    ax.set_title(col)
plt.suptitle('Numeric Features by Class (0 = no disease, 1 = disease)')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '2_numeric_features.png'))


# 3. Categorical features: what fraction of patients in each group has disease?
print("\nDisease rate per category:")
for col in CATEGORY_FEATURES:
    rate = df.groupby(col)['target'].agg(['mean', 'count'])
    rate.columns = ['disease_rate', 'patients']
    print(f"\n{col}:")
    print(rate.round(2))

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flatten(), CATEGORY_FEATURES):
    sns.barplot(data=df, x=col, y='target', errorbar=None, ax=ax)
    ax.set_ylabel('Disease rate')
    ax.set_ylim(0, 1)
    ax.set_title(col)
plt.suptitle('Disease Rate per Category')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '3_categorical_features.png'))


# 4. Correlation: which features move together with the target (and with each other)?
corr = df.corr()
print("\nCorrelation with target (strongest first):")
print(corr['target'].drop('target').sort_values(key=abs, ascending=False).round(3))

plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '4_correlation_matrix.png'))

print(f"\nPlots saved to: {PLOTS_DIR}")

# Show all plots
plt.show()
