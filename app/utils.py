import pandas as pd
import numpy as np

def load_and_preprocess(filepath="data/telco_churn.csv"):
    df = pd.read_csv(filepath)
    
    # Fix TotalCharges column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID
    df.drop('customerID', axis=1, inplace=True)
    
    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode binary columns
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 
                                'Male': 1, 'Female': 0})
    
    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

def get_raw_data(filepath="data/telco_churn.csv"):
    df = pd.read_csv(filepath)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df