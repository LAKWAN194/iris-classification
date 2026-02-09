import pandas as pd
import warnings
from IPython.display import display

warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(filepath)
    return df

def display_data_info(df):
    """
    Display basic information about the dataset
    """
    print("First few rows:")
    display(df.head())
    
    print("\nMissing values in each column:")
    print(df.isnull().sum().sort_values(ascending=False))
    
    print("\nData types:")
    print(df.info())
    
    print("\nSummary statistics:")
    print(df.describe())
    
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    if df.duplicated().sum() > 0:
        print("Duplicate rows:")
        print(df[df.duplicated()])
    
    return df