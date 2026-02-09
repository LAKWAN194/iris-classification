import pandas as pd
import numpy as np
import joblib

def load_model(model_path="../models/iris_classifier.pkl"):
    """
    Load trained model
    """
    model = joblib.load(model_path)
    return model

def make_predictions(model, X):
    """
    Make predictions using trained model
    """
    predictions = model.predict(X)
    return predictions

def predict_single_sample(model, sepal_length, sepal_width, 
                         petal_length, petal_width, 
                         class_mapping=None):
    """
    Predict class for a single sample
    """
    # Create DataFrame for single sample
    sample_data = pd.DataFrame({
        'sepal_length': [sepal_length],
        'sepal_width': [sepal_width],
        'petal_length': [petal_length],
        'petal_width': [petal_width]
    })
    
    # Make prediction
    prediction = model.predict(sample_data)[0]
    
    # Map prediction back to class name if mapping provided
    if class_mapping:
        # Reverse the mapping
        reverse_mapping = {v: k for k, v in class_mapping.items()}
        return reverse_mapping[prediction]
    
    return prediction

def batch_predict(model, input_csv_path, output_csv_path=None):
    """
    Make predictions for batch data from CSV
    """
    # Load input data
    input_data = pd.read_csv(input_csv_path)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    # Add predictions to dataframe
    result_df = input_data.copy()
    result_df['predicted_class'] = predictions
    
    # Save results if output path provided
    if output_csv_path:
        result_df.to_csv(output_csv_path, index=False)
        print(f"Predictions saved to {output_csv_path}")
    
    return result_df