import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate model performance with multiple metrics
    """
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Probabilities (needed for ROC-AUC & Log Loss)
    y_test_proba = model.predict_proba(X_test)

    # Accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Precision, Recall, F1 (macro = treats all classes equally)
    precision = precision_score(y_test, y_test_pred, average='macro')
    recall = recall_score(y_test, y_test_pred, average='macro')
    f1 = f1_score(y_test, y_test_pred, average='macro')

    # ROC-AUC (multiclass)
    roc_auc = roc_auc_score(
        y_test,
        y_test_proba,
        multi_class='ovr',
        average='macro'
    )

    # Log Loss
    logloss = log_loss(y_test, y_test_proba)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Classification Report
    report = classification_report(y_test, y_test_pred)

    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'log_loss': logloss,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_test_pred': y_test_pred,
        'y_test_proba': y_test_proba
    }


def print_evaluation_results(results):
    """
    Print evaluation results
    """
    print(f"Train Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}\n")

    print(f"Precision (Macro): {results['precision']:.4f}")
    print(f"Recall (Macro):    {results['recall']:.4f}")
    print(f"F1-score (Macro):  {results['f1_score']:.4f}\n")

    print(f"ROC-AUC (Macro):   {results['roc_auc']:.4f}")
    print(f"Log Loss:          {results['log_loss']:.4f}\n")

    print("Confusion Matrix:")
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