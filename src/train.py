import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_processed_data(X_path="../data/processed/X.csv", 
                       y_path="../data/processed/y.csv"):
    """
    Load processed features and target
    """
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_params=None):
    """
    Train logistic regression model
    """
    if model_params is None:
        model_params = {'max_iter': 200}
    
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)
    
    return model

def save_model(model, model_path="../models/iris_classifier.pkl"):
    """
    Save trained model
    """
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model_path