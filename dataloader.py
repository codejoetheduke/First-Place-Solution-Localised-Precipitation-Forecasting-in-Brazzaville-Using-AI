import numpy as np
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')

def load_data(train_path, test_path):
    """
    Loads training and testing datasets as Pandas DataFrames.
    
    Parameters:
    - train_path (str): Path to the training CSV file.
    - test_path (str): Path to the testing CSV file.
    
    Returns:
    - train_df (pd.DataFrame): Training dataset.
    - test_df (pd.DataFrame): Testing dataset.
    """
    
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"✅ Data loaded successfully!\n - Train shape: {train_df.shape}\n - Test shape: {test_df.shape}")
        return train_df, test_df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None