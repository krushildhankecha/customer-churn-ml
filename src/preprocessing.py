import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, is_train=False):
    df = df.copy()

    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    # Only map 'Churn' if in training mode and column exists
    if is_train and 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Encode all categorical columns (except 'Churn' in training)
    exclude = ['Churn'] if is_train else []
    cat_cols = df.select_dtypes(include='object').columns.difference(exclude)

    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Fill missing values if needed
    df = df.fillna(0)

    return df
