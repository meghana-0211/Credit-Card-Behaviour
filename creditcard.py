# -*- coding: utf-8 -*-
"""creditcard.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1x86A67skinsv5zpmgvzM4WjUbocyjaAk

# **Loading the data**
"""

import zipfile
with zipfile.ZipFile('/content/Dev_data_to_be_shared 3.zip','r') as zipref:
  zipref.extractall('/content/extracted')

import pandas as pd
df=pd.read_csv('/content/extracted/Dev_data_to_be_shared.csv')

"""# **Understanding the data**"""

print(df.head())

print(df.tail())

print(df.shape)

print(df.describe())

print(df.info())

print(df.isnull().sum)

for col in df.select_dtypes(include='object').columns:
  print(f'{col}: {df[col].nunique()} unique values')

print(df['bad_flag'].value_counts())

print(df.corr())

import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))
plt.show()
#nvm idek why i did this

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_credit_data(df):
    """
    Analyze credit card behavioral data and generate preprocessing insights

    Parameters:
    df (pandas.DataFrame): Input credit card data

    Returns:
    dict: Analysis results and recommendations
    """
    analysis = {}

    # Basic dataset information
    analysis['basic_info'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'target_distribution': df['bad_flag'].value_counts(normalize=True).to_dict()
    }

    # Missing value analysis
    missing_vals = df.isnull().sum()
    missing_percentages = (missing_vals / len(df)) * 100
    analysis['missing_values'] = {
        'columns_with_missing': missing_vals[missing_vals > 0].to_dict(),
        'high_missing_cols': missing_vals[missing_percentages > 30].index.tolist()
    }

    # Data type analysis
    analysis['dtypes'] = df.dtypes.value_counts().to_dict()

    # Analyze different variable groups
    var_groups = {
        'onus': [col for col in df.columns if col.startswith('onus_attribute')],
        'transaction': [col for col in df.columns if col.startswith('transaction_attribute')],
        'bureau': [col for col in df.columns if col.startswith('bureau') and not col.startswith('bureau_enquiry')],
        'bureau_enquiry': [col for col in df.columns if col.startswith('bureau_enquiry')]
    }

    # Correlation analysis for each group
    analysis['correlations'] = {}
    for group_name, columns in var_groups.items():
        if columns:
            group_df = df[columns + ['bad_flag']]
            correlations = group_df.corr()['bad_flag'].sort_values(ascending=False)
            analysis['correlations'][group_name] = correlations[correlations != 1].head(10).to_dict()

    # Zero variance analysis
    zero_var_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            zero_var_cols.append(col)
    analysis['zero_variance_columns'] = zero_var_cols

    # Identify highly correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col1, col2) for col1, col2 in zip(*np.where(upper > 0.95))]
        analysis['high_correlation_pairs'] = high_corr_pairs

    # Generate preprocessing recommendations
    analysis['preprocessing_recommendations'] = {
        'handle_missing': [
            f"Columns with >30% missing values: Consider dropping or using advanced imputation",
            "For columns with <30% missing values: Use median/mode imputation based on data distribution"
        ],
        'feature_engineering': [
            "Create ratio features for transaction amounts",
            "Calculate utilization ratios from credit limits",
            "Generate time-based features from transaction patterns",
            "Create aggregated bureau metrics"
        ],
        'scaling_encoding': [
            "Use robust scaler for amount-based features to handle outliers",
            "Apply log transformation for highly skewed numerical features",
            "One-hot encode categorical variables if present"
        ]
    }

    return analysis

df = pd.read_csv('/content/extracted/Dev_data_to_be_shared.csv')
analysis_results = analyze_credit_data(df)

print(analysis_results)

"""# **Preprocessing the data**"""

df=df.dropna(how='all')

df = df.dropna(axis=1, how='all')

median=df.median()

df = df.fillna(median)

print(df.isnull().sum())

variances = df.var()

zero_variance_features=variances[variances==0].index

df=df.drop(columns=zero_variance_features)

print(df.shape)



import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

def preprocess_credit_data(df):
    """
    Preprocess credit card data based on analysis insights

    Parameters:
    df (pandas.DataFrame): Input credit card data

    Returns:
    tuple: (preprocessed_df, feature_info)
    """
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    feature_info = {}

    # 1. Handle Missing Values
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='constant', fill_value=0)

    # Separate numeric and categorical columns
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
    categorical_cols = processed_df.select_dtypes(exclude=[np.number]).columns

    # Impute values
    processed_df[numeric_cols] = numeric_imputer.fit_transform(processed_df[numeric_cols])
    if len(categorical_cols) > 0:
        processed_df[categorical_cols] = categorical_imputer.fit_transform(processed_df[categorical_cols])

    # 2. Feature Engineering
    new_features = {}

    # Transaction ratios
    transaction_cols = [col for col in processed_df.columns if col.startswith('transaction_attribute')]
    if len(transaction_cols) >= 2:
        for i in range(len(transaction_cols)-1):
            for j in range(i+1, len(transaction_cols)):
                col1, col2 = transaction_cols[i], transaction_cols[j]
                if processed_df[col2].mean() != 0:
                    ratio_name = f'ratio_{i}_{j}'
                    new_features[ratio_name] = processed_df[col1] / (processed_df[col2] + 1e-6)
                    feature_info[ratio_name] = f'Ratio of {col1} to {col2}'

    # Bureau aggregates
    bureau_cols = [col for col in processed_df.columns if col.startswith('bureau_') and not col.startswith('bureau_enquiry')]
    if bureau_cols:
        new_features['bureau_mean'] = processed_df[bureau_cols].mean(axis=1)
        new_features['bureau_std'] = processed_df[bureau_cols].std(axis=1)
        new_features['bureau_max'] = processed_df[bureau_cols].max(axis=1)
        feature_info['bureau_aggregates'] = 'Statistical aggregates of bureau features'

    # Add all new features at once using concat
    if new_features:
        new_features_df = pd.DataFrame(new_features)
        processed_df = pd.concat([processed_df, new_features_df], axis=1)

    # 3. Handle Skewed Features
    numeric_features = processed_df.select_dtypes(include=[np.number]).columns
    log_transformed_features = {}

    for col in numeric_features:
        skewness = stats.skew(processed_df[col])
        if abs(skewness) > 1.5:  # High skewness threshold
            log_transformed_features[f'{col}_log'] = np.log1p(processed_df[col] - processed_df[col].min() + 1)
            feature_info[f'{col}_log'] = f'Log transformed {col} due to high skewness'

    # Add log-transformed features at once
    if log_transformed_features:
        log_features_df = pd.DataFrame(log_transformed_features)
        processed_df = pd.concat([processed_df, log_features_df], axis=1)

    # 4. Scale Features
    scaler = RobustScaler()
    amount_cols = [col for col in numeric_features if 'amount' in col.lower() or 'transaction' in col.lower()]
    if amount_cols:
        processed_df[amount_cols] = scaler.fit_transform(processed_df[amount_cols])
        feature_info['scaled_amounts'] = 'Amount features scaled using RobustScaler'

    # 5. Remove Zero Variance Features
    zero_var_cols = [col for col in processed_df.columns if processed_df[col].nunique() <= 1]
    processed_df = processed_df.drop(columns=zero_var_cols)
    feature_info['removed_features'] = f'Removed {len(zero_var_cols)} zero variance features'

    return processed_df, feature_info

def generate_preprocessing_report(original_df, processed_df, feature_info):
    """
    Generate a report comparing original and preprocessed data

    Parameters:
    original_df (pandas.DataFrame): Original dataframe
    processed_df (pandas.DataFrame): Preprocessed dataframe
    feature_info (dict): Information about feature transformations

    Returns:
    dict: Report containing preprocessing insights
    """
    report = {
        'original_shape': original_df.shape,
        'processed_shape': processed_df.shape,
        'new_features': len(processed_df.columns) - len(original_df.columns),
        'removed_features': len(original_df.columns) - len(processed_df.columns),
        'feature_transformations': feature_info,
        'memory_usage': {
            'original': original_df.memory_usage().sum() / 1024**2,  # MB
            'processed': processed_df.memory_usage().sum() / 1024**2  # MB
        }
    }

    return report

# Example usage
if __name__ == "__main__":

    # Preprocess data
    processed_df, feature_info = preprocess_credit_data(df)

    # Generate report
    report = generate_preprocessing_report(df, processed_df, feature_info)

    # Print key insights
    print("Preprocessing Complete!")
    print(f"Original shape: {report['original_shape']}")
    print(f"Processed shape: {report['processed_shape']}")
    print(f"New features added: {report['new_features']}")
    print(f"Memory usage reduced by: {(report['memory_usage']['original'] - report['memory_usage']['processed']):.2f} MB")