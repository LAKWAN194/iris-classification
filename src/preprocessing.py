import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

def check_data_quality(df):
    """
    Perform data quality checks
    """
    # Check for missing values
    if df.isna().any().any():
        raise ValueError("Missing values detected in dataset")
    else:
        print("No missing values detected")
    
    return df

def encode_categorical(df, target_column='class'):
    """
    Encode categorical variables
    """
    df_encoded = df.copy()
    le = LabelEncoder()
    df_encoded['class_encoded'] = le.fit_transform(df_encoded[target_column])
    
    # Print class mapping
    original_classes = df[target_column].unique()
    encoded_classes = le.classes_
    class_mapping = dict(zip(encoded_classes, range(len(encoded_classes))))
    print("Class mapping:", class_mapping)
    
    return df_encoded, le

def prepare_features_targets(df_encoded, feature_columns=None, target_column='class_encoded'):
    """
    Prepare features (X) and target (y)
    """
    if feature_columns is None:
        feature_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    X = df_encoded[feature_columns]
    y = df_encoded[target_column]
    
    return X, y

def create_correlation_heatmap(X, figsize=(8, 6)):
    """
    Create correlation heatmap for features
    """
    plt.figure(figsize=figsize)
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    
    return X.corr()

def save_processed_data(X, y, X_path="../data/processed/X.csv", 
                       y_path="../data/processed/y.csv"):
    """
    Save processed features and target
    """
    # Save features
    X.to_csv(X_path, index=False)
    
    # Save target
    pd.DataFrame(y, columns=["class_encoded"]).to_csv(y_path, index=False)
    
    print(f"Features saved to {X_path}")
    print(f"Target saved to {y_path}")