"""
Iris Classification Project
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .data_loader import load_data, display_data_info
from .preprocessing import (check_data_quality, encode_categorical, 
                           prepare_features_targets, create_correlation_heatmap,
                           save_processed_data)
from .train import load_processed_data, split_data, train_model, save_model
from .evaluate import evaluate_model, print_evaluation_results, plot_confusion_matrix