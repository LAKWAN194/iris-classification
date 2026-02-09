import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Generate classification report
    report = classification_report(y_test, y_test_pred)
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_test_pred': y_test_pred
    }

def print_evaluation_results(results):
    """
    Print evaluation results
    """
    print("Train Accuracy:", results['train_accuracy'])
    print("Test Accuracy:", results['test_accuracy'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])

def plot_confusion_matrix(cm, class_names=None, figsize=(8, 6)):
    """
    Plot confusion matrix as heatmap
    """
    if class_names is None:
        class_names = ['Setosa', 'Versicolor', 'Virginica']
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()