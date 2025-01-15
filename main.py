import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.base import clone
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import zipfile
import os
import tempfile
warnings.filterwarnings('ignore')

df_dev = pd.read_csv("Dev_data_to_be_shared.csv")
df_val = pd.read_csv("validation_data_to_be_shared.csv")

df_dev.info()


df_dev.describe()

df_val.describe()

"""ok, so the account numbers for the dev data are from 1 to 5855 and the account numbers for validation dataset are from 1,00,001 to 1,05,882 (this info might not be useful)

let's check if any of the columns in any of the datasets is all 0s
"""

all_zeros_columns = df_dev.columns[(df_dev == 0).all()]
all_zeros_columns

all_zeros_columns = df_val.columns[(df_val == 0).all()]
all_zeros_columns.tolist()

"""let's check if any of the rows are all 0s"""

all_zeros_rows_dev = df_dev.index[(df_dev == 0).all(axis=1)]
all_zeros_rows_dev.tolist()

all_zeros_rows_val = df_val.index[(df_val == 0).all(axis=1)]
all_zeros_rows_val.tolist()

all_zeros_rows_dev = df_dev.index[(df_dev.iloc[:, 1:] == 0).all(axis=1)]
all_zeros_rows_dev.tolist()

all_zeros_rows_val = df_val.index[(df_val.iloc[:, 1:] == 0).all(axis=1)]
all_zeros_rows_val.tolist()

"""bruh, let's try columns"""

all_zeros_columns_dev = df_dev.iloc[:, 1:].columns[(df_dev.iloc[:, 1:] == 0).all(axis=0)]
all_zeros_columns_dev.tolist()

all_zeros_columns_val = df_val.iloc[:, 1:].columns[(df_val.iloc[:, 1:] == 0).all(axis=0)]
all_zeros_columns_val.tolist()

df_dev.isnull().sum(), df_val.isnull().sum()

df_dev.shape, df_val.shape

"""drop columns with all missing values in both datasets

create a copy, _cleaned
"""

df_dev_cleaned = df_dev.dropna(axis=1, how='all')
df_val_cleaned = df_val.dropna(axis=1, how='all')

df_dev_cleaned.isnull().sum(), df_val_cleaned.isnull().sum()

df_dev_cleaned.shape, df_val_cleaned.shape

"""drop columns with more than 50% missing values"""

threshold = 0.5
missing_percentage_dev = df_dev_cleaned.isnull().mean()
columns_with_missing_data = missing_percentage_dev[missing_percentage_dev > threshold]
columns_with_missing_data

df_dev_cleaned = df_dev_cleaned.loc[:, missing_percentage_dev < threshold]

threshold = 0.5
missing_percentage_val = df_val_cleaned.isnull().mean()

df_val_cleaned = df_val_cleaned.loc[:, missing_percentage_val < threshold]

df_dev_cleaned.shape, df_val_cleaned.shape

"""check how balanced or imbalanced the bad_flag variable is"""

import matplotlib.pyplot as plt
import seaborn as sns

"""Plot the distribution of the target variable 'bad_flag' for df_dev_cleaned"""

plt.figure(figsize=(6, 4))
sns.countplot(x='bad_flag', data=df_dev_cleaned)
plt.title('Distribution of bad_flag')
plt.show()

bad_flag_counts = df_dev_cleaned['bad_flag'].value_counts()
bad_flag_counts

"""Exploring onus attributes"""

onus_columns = [col for col in df_dev_cleaned.columns if 'onus_attribute' in col]
df_dev_cleaned[onus_columns].describe()

summary_stats = {}

for column in onus_columns:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()

    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

"""IQR, hmmm, calculate IQR for each 'onus_attribute_x'"""

for column in df_dev_cleaned.columns:
    if column.startswith("onus_attribute"):
        Q1 = df_dev_cleaned[column].quantile(0.25)
        Q3 = df_dev_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

"""before removing outliners, plot a histogram"""

columns_to_plot = ['onus_attribute_1', 'onus_attribute_2', 'onus_attribute_3', 'onus_attribute_4', 'onus_attribute_19']
plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, len(columns_to_plot)//2 + 1, i)
    sns.histplot(df_dev_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} Before Outlier Removal')

plt.tight_layout()
plt.show()

"""now remove otliners and then plot a histogram

"""

for column in columns_to_plot:
    Q1 = df_dev_cleaned[column].quantile(0.25)
    Q3 = df_dev_cleaned[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    df_dev_cleaned = df_dev_cleaned[(df_dev_cleaned[column] >= lower_bound) & (df_dev_cleaned[column] <= upper_bound)]

plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, len(columns_to_plot)//2 + 1, i)
    sns.histplot(df_dev_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} After Outlier Removal')

plt.tight_layout()
plt.show()

for column in df_dev_cleaned.columns:
    if column.startswith("onus_attribute"):
        Q1 = df_dev_cleaned[column].quantile(0.25)
        Q3 = df_dev_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

        df_dev_cleaned = df_dev_cleaned[(df_dev_cleaned[column] >= lower_bound) & (df_dev_cleaned[column] <= upper_bound)]

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

onus_columns = [col for col in df_dev_cleaned.columns if 'onus_attribute' in col]
df_dev_cleaned[onus_columns].describe()

"""check extremes in onus for normalization"""

summary_stats = {}

for column in onus_columns:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()

    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

"""remove cols with all 0s"""

df_dev_cleaned = df_dev_cleaned.loc[:, (df_dev_cleaned != 0).any(axis=0)]

df_dev_cleaned

remaining_onus_attributes = [col for col in df_dev_cleaned.columns if 'onus_attribute' in col]
remaining_onus_attributes

summary_stats = {}

for column in remaining_onus_attributes:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()
    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

"""normalization of the remaining onus attributes"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

df_dev_cleaned[remaining_onus_attributes] = scaler.fit_transform(df_dev_cleaned[remaining_onus_attributes])

summary_stats_normalized = df_dev_cleaned[remaining_onus_attributes].describe()
print(summary_stats_normalized)

fig, axes = plt.subplots(nrows=len(remaining_onus_attributes), ncols=2, figsize=(12, 3*len(columns_to_plot)))


for i, column in enumerate(remaining_onus_attributes):
    # Plot Histogram
    sns.histplot(df_dev_cleaned[column], kde=True, ax=axes[i, 0], color='blue')
    axes[i, 0].set_title(f'{column} Histogram')

    # Plot Boxplot
    sns.boxplot(x=df_dev_cleaned[column], ax=axes[i, 1], color='lightgreen')
    axes[i, 1].set_title(f'{column} Boxplot')

plt.tight_layout()
plt.show()

"""let's do the same with transaction attributes"""

transaction_columns = [col for col in df_dev_cleaned.columns if 'transaction_attribute' in col]
df_dev_cleaned[transaction_columns].describe()

summary_stats = {}

for column in transaction_columns:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()

    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

""" calculate IQR for each 'transaction_attribute_x'"""

for column in df_dev_cleaned.columns:
    if column.startswith("transaction_attribute"):
        Q1 = df_dev_cleaned[column].quantile(0.25)
        Q3 = df_dev_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

columns_to_plot = ['transaction_attribute_1', 'transaction_attribute_10', 'transaction_attribute_13', 'transaction_attribute_19', 'transaction_attribute_71']
plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, len(columns_to_plot)//2 + 1, i)
    sns.histplot(df_dev_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} Before Outlier Removal')

plt.tight_layout()
plt.show()

for column in df_dev_cleaned.columns:
    if column.startswith("transaction_attribute"):
        Q1 = df_dev_cleaned[column].quantile(0.25)
        Q3 = df_dev_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

        median_value = df_dev_cleaned[column].median()
        df_dev_cleaned[column] = df_dev_cleaned[column].apply(
            lambda x: median_value if x < lower_bound or x > upper_bound else x
        )

df_dev_cleaned.head()

plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, len(columns_to_plot)//2 + 1, i)
    sns.histplot(df_dev_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} After Outlier Removal')

plt.tight_layout()
plt.show()

transaction_attributes = [col for col in df_dev_cleaned.columns if 'transaction_attributes' in col]
df_dev_cleaned[transaction_columns].describe()

summary_stats = {}

for column in transaction_columns:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()

    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

df_dev_cleaned = df_dev_cleaned.loc[:, (df_dev_cleaned != 0).any(axis=0)]

df_dev_cleaned

remaining_transaction_attributes = [col for col in df_dev_cleaned.columns if 'transaction_attribute' in col]
remaining_transaction_attributes

summary_stats = {}

for column in remaining_transaction_attributes:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()
    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

df_dev_cleaned[remaining_transaction_attributes] = scaler.fit_transform(df_dev_cleaned[remaining_transaction_attributes])

summary_stats_normalized = df_dev_cleaned[remaining_transaction_attributes].describe()
print(summary_stats_normalized)


df_dev_cleaned.describe()

bureau_enquiry_columns = [col for col in df_dev_cleaned.columns if 'bureau_enquiry' in col]
df_dev_cleaned[bureau_enquiry_columns].describe()

summary_stats = {}

for column in bureau_enquiry_columns:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()

    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

for column in df_dev_cleaned.columns:
    if column.startswith("bureau_enquiry"):
        Q1 = df_dev_cleaned[column].quantile(0.25)
        Q3 = df_dev_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

columns_to_plot = ['bureau_enquiry_1', 'bureau_enquiry_2', 'bureau_enquiry_7', 'bureau_enquiry_9', 'bureau_enquiry_48']
plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, len(columns_to_plot)//2 + 1, i)
    sns.histplot(df_dev_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} Before Outlier Removal')

plt.tight_layout()
plt.show()

for column in df_dev_cleaned.columns:
    if column.startswith("bureau_enquiry"):
        Q1 = df_dev_cleaned[column].quantile(0.25)
        Q3 = df_dev_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 3.0 * IQR
        upper_bound = Q3 + 3.0 * IQR

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

        median_value = df_dev_cleaned[column].median()
        df_dev_cleaned[column] = df_dev_cleaned[column].apply(
            lambda x: median_value if x < lower_bound or x > upper_bound else x
        )

        outliers = df_dev_cleaned[(df_dev_cleaned[column] < lower_bound) | (df_dev_cleaned[column] > upper_bound)]
        print(f"Outliers detected in {column}: {len(outliers)}")

df_dev_cleaned.head()

plt.figure(figsize=(12, 8))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(2, len(columns_to_plot)//2 + 1, i)
    sns.histplot(df_dev_cleaned[column], kde=True)
    plt.title(f'Distribution of {column} After Outlier Removal')

plt.tight_layout()
plt.show()

bureau_enquiry_columns = [col for col in df_dev_cleaned.columns if 'bureau_enquiry' in col]
df_dev_cleaned[bureau_enquiry_columns].describe()

summary_stats = {}

for column in bureau_enquiry_columns:
    min_value = df_dev_cleaned[column].min()
    max_value = df_dev_cleaned[column].max()
    avg_value = df_dev_cleaned[column].mean()

    summary_stats[column] = {'min': min_value, 'max': max_value, 'avg': avg_value}

summary_stats_df = pd.DataFrame(summary_stats).T

summary_stats_df

df_dev_cleaned = df_dev_cleaned.loc[:, (df_dev_cleaned != 0).any(axis=0)]

df_dev_cleaned

remaining_bureau_enquiries = [col for col in df_dev_cleaned.columns if 'bureau_enquiry' in col]
remaining_bureau_enquiries

bad_flag_counts = df_dev_cleaned['bad_flag'].value_counts()
bad_flag_counts

"""lol, itni mehnat aur ye nateeja"""

bad_flag_rows = df_dev[df_dev['bad_flag'] == 1.0]
bad_flag_rows

numeric_columns = bad_flag_rows.select_dtypes(include=['float64', 'int64']).columns

bad_flag_rows[numeric_columns] = scaler.fit_transform(bad_flag_rows[numeric_columns])

df_dev_cleaned = pd.concat([df_dev_cleaned, bad_flag_rows], ignore_index=True)

print(f"Rows in df_dev_cleaned after appending: {len(df_dev_cleaned)}")
df_dev_cleaned

bad_flag_counts = df_dev_cleaned['bad_flag'].value_counts()
bad_flag_counts

# Step 1: Filter the rows where bad_flag == 1 from df_dev
bad_flag_rows = df_dev[df_dev['bad_flag'] == 1]

# Step 2: Normalize the numerical columns of the bad_flag rows
# Assuming you're using MinMaxScaler or a similar method for normalization
from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
scaler = MinMaxScaler()

# Apply the scaler to all columns except 'bad_flag' (which should not be normalized)
columns_to_normalize = bad_flag_rows.drop(columns=['bad_flag'])
normalized_rows = pd.DataFrame(scaler.fit_transform(columns_to_normalize), columns=columns_to_normalize.columns)

# Add back the 'bad_flag' column
normalized_rows['bad_flag'] = bad_flag_rows['bad_flag'].values

# Step 3: Append the normalized rows to df_dev_cleaned
df_dev_cleaned = pd.concat([df_dev_cleaned, normalized_rows], ignore_index=True)

# Verify the new shape of the dataframe
print(df_dev_cleaned.shape)

bad_flag_counts = df_dev_cleaned['bad_flag'].value_counts()
bad_flag_counts

df_dev_cleaned

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_dev_cleaned_imputed = pd.DataFrame(imputer.fit_transform(df_dev_cleaned), columns=df_dev_cleaned.columns)

missing_columns = df_dev_cleaned.columns[df_dev_cleaned.isna().all()]
print(missing_columns)

df_dev_cleaned = df_dev_cleaned.drop(columns=['bureau_436', 'bureau_447'])

imputer = KNNImputer(n_neighbors=5)
df_dev_cleaned_imputed = pd.DataFrame(imputer.fit_transform(df_dev_cleaned), columns=df_dev_cleaned.columns)

df_dev_cleaned_imputed

bad_flag_counts = df_dev_cleaned_imputed['bad_flag'].value_counts()
bad_flag_counts

import numpy as np
import pandas as pd

df = pd.read_csv("cleaned_dev.csv")


df_1 = df[df["bad_flag"] == 1.0]
df_0 = df[df["bad_flag"] == 0.0]


df_0 = df_0.sample(frac=1, random_state=42).reset_index(drop=True)

splits = np.array_split(df_0, 6)


dfs = [pd.concat([df_1, split]).sample(frac=1, random_state=42).reset_index(drop=True) for split in splits]

output_file = "balanced_datasets.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for i, df in enumerate(dfs):
        sheet_name = f"df_{i + 1}"
        df.to_excel(writer, index=False, sheet_name=sheet_name)

from google.colab import files

df_dev_cleaned_imputed.to_csv('cleaned_dev.csv', index=False)
files.download('cleaned_dev.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -*- coding: utf-8 -*-
"""ensemble.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1x7RAweT0YsnwOt63xpTXqW0d8dKNK4gY
"""



def extract_zip(zip_path, extract_dir=None):
    """
    Extract a ZIP file and return the path to the CSV file inside.

    Parameters:
    zip_path (str): Path to the ZIP file
    extract_dir (str): Directory to extract files to. If None, uses a temporary directory

    Returns:
    str: Path to the extracted CSV file
    """
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

        # Find the CSV file in the extracted contents
        csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV file found in the ZIP archive")

        return os.path.join(extract_dir, csv_files[0])

class BalancedEnsembleCreditScorer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names_ = None
        self.training_features_ = None

        # Define base models with provided parameters
        self.base_models = {
            'xgb1': xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.02, max_depth=4,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=4,
                scale_pos_weight=3, random_state=random_state
            ),
            'xgb2': xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.03, max_depth=5,
                subsample=0.85, colsample_bytree=0.9, min_child_weight=6,
                scale_pos_weight=3, random_state=random_state
            ),
            'xgb3': xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=3,
                subsample=0.9, colsample_bytree=0.75, min_child_weight=3,
                scale_pos_weight=3, random_state=random_state
            ),
            'rf1': RandomForestClassifier(
                n_estimators=500, max_depth=5, min_samples_split=25,
                min_samples_leaf=10, class_weight={0: 1, 1: 3},
                random_state=random_state
            ),
            'rf2': RandomForestClassifier(
                n_estimators=500, max_depth=6, min_samples_split=20,
                min_samples_leaf=8, class_weight={0: 1, 1: 3},
                random_state=random_state
            ),
            'rf3': RandomForestClassifier(
                n_estimators=500, max_depth=4, min_samples_split=30,
                min_samples_leaf=12, class_weight={0: 1, 1: 3},
                random_state=random_state
            )
        }

        # Initialize meta-model
        self.meta_model = LogisticRegression(
            C=0.8,
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )

        # Initialize containers for trained models
        self.trained_models = {i: {} for i in range(1, 7)}  # One dict per balanced dataset

    def engineer_features(self, X):
        """Engineer features from transaction, bureau, and onus attributes"""
        X_new = X.copy()

        # Transaction attribute aggregations
        tx_cols = [col for col in X.columns if col.startswith('transaction_attribute_')]
        if tx_cols:
            X_new['avg_nonzero_tx'] = X[tx_cols].replace(0, np.nan).mean(axis=1)
            X_new['nonzero_tx_count'] = (X[tx_cols] != 0).sum(axis=1)
            X_new['tx_volatility'] = X[tx_cols].std(axis=1)

        # Bureau data aggregations
        bureau_cols = [col for col in X.columns if col.startswith('bureau_')]
        if bureau_cols:
            X_new['avg_bureau_score'] = X[bureau_cols].mean(axis=1)
            X_new['bureau_score_spread'] = X[bureau_cols].std(axis=1)
            X_new['active_bureau_count'] = (X[bureau_cols] != 0).sum(axis=1)

        # Onus attribute aggregations
        onus_cols = [col for col in X.columns if col.startswith('onus_attribute_')]
        if onus_cols:
            X_new['avg_onus_score'] = X[onus_cols].mean(axis=1)
            X_new['onus_volatility'] = X[onus_cols].std(axis=1)

        # Bureau enquiry features
        enquiry_cols = [col for col in X.columns if col.startswith('bureau_enquiry_')]
        if enquiry_cols:
            X_new['total_enquiries'] = X[enquiry_cols].sum(axis=1)
            recent_enquiry_cols = enquiry_cols[-10:]
            X_new['recent_enquiry_intensity'] = X[recent_enquiry_cols].sum(axis=1)

        # Interaction features
        if 'avg_bureau_score' in X_new.columns and 'total_enquiries' in X_new.columns:
            X_new['bureau_enquiry_risk'] = X_new['avg_bureau_score'] / (X_new['total_enquiries'] + 1)

        # Fill NaN values and store feature names
        X_new = X_new.fillna(0)
        self.feature_names_ = list(X_new.columns)

        return X_new

    def _get_base_predictions(self, X_scaled):
        """Get predictions from all base models"""
        predictions = np.zeros((X_scaled.shape[0], len(self.trained_models) * len(self.base_models)))
        col_idx = 0

        for dataset_idx in self.trained_models:
            for model_name, model in self.trained_models[dataset_idx].items():
                predictions[:, col_idx] = model.predict_proba(X_scaled)[:, 1]
                col_idx += 1

        return predictions

    def fit(self, balanced_datasets):
        """
        Fit models on multiple balanced datasets and train meta-model

        Parameters:
        balanced_datasets: dict of pandas DataFrames, keys should be 1-6
        """
        print("Training base models on balanced datasets...")

        # Store feature names from first dataset
        X_first = balanced_datasets[1].drop(['bad_flag', 'account_number'], axis=1)
        X_engineered = self.engineer_features(X_first)
        self.training_features_ = self.feature_names_

        # Initialize array to store meta-features
        meta_features = []
        meta_labels = []

        # Train models on each balanced dataset
        for dataset_idx, df in balanced_datasets.items():
            print(f"\nTraining models on balanced dataset {dataset_idx}")

            # Prepare features
            X = df.drop(['bad_flag', 'account_number'], axis=1)
            y = df['bad_flag']

            # Engineer and scale features
            X_engineered = self.engineer_features(X)
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_engineered),
                columns=self.feature_names_,
                index=X_engineered.index
            )

            # Train each base model
            for model_name, model in self.base_models.items():
                print(f"Training {model_name} on dataset {dataset_idx}...")
                model_clone = clone(model)
                model_clone.fit(X_scaled, y)
                self.trained_models[dataset_idx][model_name] = model_clone

            # Get predictions for meta-model training
            dataset_predictions = self._get_base_predictions(X_scaled)
            meta_features.append(dataset_predictions)
            meta_labels.append(y)

        # Combine all meta-features and train meta-model
        print("\nTraining meta-model...")
        X_meta = np.vstack(meta_features)
        y_meta = np.concatenate(meta_labels)
        self.meta_model.fit(X_meta, y_meta)

        return self

    def predict_proba(self, X):
        """Generate predictions using meta-model combination of base models"""
        # Engineer features
        X_engineered = self.engineer_features(X)
        X_engineered = X_engineered[self.training_features_]

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_engineered),
            columns=self.training_features_,
            index=X_engineered.index
        )

        # Get base model predictions
        base_predictions = self._get_base_predictions(X_scaled)

        # Generate final predictions using meta-model
        final_predictions = self.meta_model.predict_proba(base_predictions)[:, 1]

        return final_predictions

    def evaluate(self, X, y):
        """Evaluate model performance with multiple metrics"""
        # Generate predictions
        y_pred_proba = self.predict_proba(X)

        # Find optimal threshold using precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_threshold = thresholds[np.argmax(f1_scores[:-1])]

        # Generate binary predictions using optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        metrics = {
            'threshold': optimal_threshold,
            'accuracy': accuracy_score(y, y_pred),
            'auc_score': roc_auc_score(y, y_pred_proba),
            'avg_precision': average_precision_score(y, y_pred_proba),
            'confusion_matrix': {
                'tn': tn, 'fp': fp,
                'fn': fn, 'tp': tp
            }
        }

        # Print detailed metrics
        print("\nModel Performance Metrics:")
        print("=" * 50)
        print(f"Optimal Threshold: {optimal_threshold:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"AUC Score: {metrics['auc_score']:.4f}")
        print(f"Average Precision: {metrics['avg_precision']:.4f}")

        print("\nConfusion Matrix:")
        print("-" * 50)
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives: {tp}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nAdditional Metrics:")
        print("-" * 50)
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return metrics

from sklearn.model_selection import train_test_split
import pandas as pd
import tempfile

# Example usage
if __name__ == "__main__":
    # Create temporary directory for zip extraction
    temp_dir = tempfile.mkdtemp()

    # Load and split balanced datasets
    balanced_datasets_train = {}
    balanced_datasets_test = {}

    print("Loading and splitting datasets...")
    for i in range(1, 7):
        # Load dataset
        df = pd.read_excel('/content/balanced_datasets (1).xlsx', sheet_name=f'df_{i}')

        # Split the data
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42,
            stratify=df['bad_flag']
        )

        balanced_datasets_train[i] = train_df
        balanced_datasets_test[i] = test_df

    # Initialize and train model
    print("\nTraining model...")
    model = BalancedEnsembleCreditScorer()
    model.fit(balanced_datasets_train)

    # Combine all test sets and remove duplicates
    print("\nPreparing test data...")
    all_test_data = pd.concat([df for df in balanced_datasets_test.values()], axis=0)
    all_test_data = all_test_data.drop_duplicates(subset='account_number')

    # Evaluate model
    print("\nEvaluating model on test set...")
    X_test = all_test_data.drop(['bad_flag', 'account_number'], axis=1)
    y_test = all_test_data['bad_flag']

    # Get evaluation metrics
    metrics = model.evaluate(X_test, y_test)

    # Generate and save predictions for test set
    predictions = pd.DataFrame({
        'account_number': all_test_data['account_number'],
        'actual': y_test,
        'predicted_probability': model.predict_proba(X_test)
    })
    predictions.to_csv('predictions_balanced_ensemble101.csv', index=False)

    # Load and process validation data
    print("\nProcessing validation data...")
    val_csv_path = extract_zip('/content/validation_data_to_be_shared 3.zip', temp_dir)
    val_data = pd.read_csv(val_csv_path)

    # Generate predictions for validation data
    print("Generating predictions for validation data...")
    val_predictions = model.predict_proba(val_data.drop(['account_number'], axis=1))

    # Create submission file
    submission = pd.DataFrame({
        'account_number': val_data['account_number'],
        'predicted_probability': val_predictions
    })

    # Save validation predictions
    print("Saving validation predictions...")
    submission.to_csv('validation_predictions.csv', index=False)

    print("\nCompleted! Files saved:")
    print("1. predictions_balanced_ensemble101.csv - Test set predictions")
    print("2. validation_predictions.csv - Validation set predictions")

