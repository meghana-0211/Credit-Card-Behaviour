# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LuFimIvlJdC4cEadtHicVkk__9yNO_df

I did not make a copy. This is a separate notebook
"""

import pandas as pd

df_dev = pd.read_csv("Dev_data_to_be_shared.csv")
df_val = pd.read_csv("validation_data_to_be_shared.csv")

df_dev.info()

"""welp, the datsets:"""

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

"""nvm

oh wait, the account number is never 0, identify rows with all 0s excluding the account number, maybe?
"""

all_zeros_rows_dev = df_dev.index[(df_dev.iloc[:, 1:] == 0).all(axis=1)]
all_zeros_rows_dev.tolist()

all_zeros_rows_val = df_val.index[(df_val.iloc[:, 1:] == 0).all(axis=1)]
all_zeros_rows_val.tolist()

"""bruh, let's try columns"""

all_zeros_columns_dev = df_dev.iloc[:, 1:].columns[(df_dev.iloc[:, 1:] == 0).all(axis=0)]
all_zeros_columns_dev.tolist()

all_zeros_columns_val = df_val.iloc[:, 1:].columns[(df_val.iloc[:, 1:] == 0).all(axis=0)]
all_zeros_columns_val.tolist()

"""😭 nvm"""

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

# fig, axes = plt.subplots(nrows=len(remaining_transaction_attributes), ncols=2, figsize=(12, 3*len(columns_to_plot)))


# for i, column in enumerate(remaining_transaction_attributes):
#     # Plot Histogram
#     sns.histplot(df_dev_cleaned[column], kde=True, ax=axes[i, 0], color='blue')
#     axes[i, 0].set_title(f'{column} Histogram')

#     # Plot Boxplot
#     sns.boxplot(x=df_dev_cleaned[column], ax=axes[i, 1], color='lightgreen')
#     axes[i, 1].set_title(f'{column} Boxplot')

# plt.tight_layout()
# plt.show()

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

