"""
Main pipeline script to run the entire ML workflow
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, display_data_info
from preprocessing import (check_data_quality, encode_categorical, 
                          prepare_features_targets, save_processed_data)
from train import load_processed_data, split_data, train_model, save_model
from evaluate import evaluate_model, print_evaluation_results

def main():
    print("=" * 60)
    print("IRIS CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and explore data
    print("\n1. Loading data...")
    df = load_data("data/raw/iris.csv")
    display_data_info(df)
    
    # Step 2: Preprocess
    print("\n2. Preprocessing data...")
    df = check_data_quality(df)
    df_encoded, label_encoder = encode_categorical(df)
    X, y = prepare_features_targets(df_encoded)
    
    # Save cleaned data
    save_processed_data(X, y, 
                       X_path="data/processed/X.csv",
                       y_path="data/processed/y.csv")
    
    # Step 3: Train model
    print("\n3. Training model...")
    X, y = load_processed_data(X_path="data/processed/X.csv",
                              y_path="data/processed/y.csv")
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    save_model(model, model_path="models/iris_classifier.pkl")
    
    # Step 4: Evaluate
    print("\n4. Evaluating model...")
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    print_evaluation_results(results)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()