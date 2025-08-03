import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold, GroupKFold
import statistics
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesFeatureEngineer:
    def __init__(self, 
                 date_col='DATE', 
                 target_col='Target',
                 diff_cols=None,
                 n_lags=2):
        """
        A class for creating lag and difference features in a combined train/test dataset.

        Parameters:
        - date_col: str, name of the date column
        - target_col: str, name of the target column
        - diff_cols: list, columns to compute first-order differences for
        - n_lags: int, number of lag/lead features to create
        """
        self.date_col = date_col
        self.target_col = target_col
        self.diff_cols = diff_cols if diff_cols else []
        self.n_lags = n_lags

    def transform(self, train_df, test_df):
        # Sort and tag
        train_df = train_df.sort_values(self.date_col).reset_index(drop=True)
        test_df = test_df.sort_values(self.date_col).reset_index(drop=True)
        train_df['Set'] = 'train'
        test_df['Set'] = 'test'
        
        # Combine datasets
        dataset = pd.concat([train_df, test_df], axis=0).sort_values(self.date_col).reset_index(drop=True)

        # Lag and lead features
        for lag in range(1, self.n_lags + 1):
            dataset[f'{self.target_col}_Lag_{lag}'] = dataset[self.target_col].shift(lag)
            dataset[f'{self.target_col}_Lead_{lag}'] = dataset[self.target_col].shift(-lag)

        # First-order difference features
        for col in self.diff_cols:
            dataset[f'{col}_diff1'] = dataset[col].diff()

        # Split back into train and test
        train_processed = dataset[dataset['Set'] == 'train'].copy()
        test_processed = dataset[dataset['Set'] == 'test'].copy()

        # Drop helper column
        train_processed.drop(columns=['Set'], inplace=True)
        test_processed.drop(columns=['Set'], inplace=True)
        
        return train_processed, test_processed


def prepare_features(train_df, test_df, target_col='Target', drop_cols=None):
    """
    Prepares X, y, and X_test by dropping specified columns.
    
    Parameters:
    - train_df: pd.DataFrame, training dataset
    - test_df: pd.DataFrame, test dataset
    - target_col: str, name of target column
    - drop_cols: list, columns to drop (in addition to target)
    
    Returns:
    - X: pd.DataFrame, training features
    - y: pd.Series, training target
    - X_test: pd.DataFrame, test features
    """
    if drop_cols is None:
        drop_cols = []
    
    # Ensure target is in drop columns
    cols_to_drop = list(set(drop_cols + [target_col]))

    # Extract target
    y = train_df[target_col]

    # Drop unnecessary columns
    X = train_df.drop(columns=cols_to_drop, errors='ignore')
    X_test = test_df[X.columns]  # Ensure same columns as X

    return X, y, X_test
