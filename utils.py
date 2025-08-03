from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
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

def ensemble_predictions(pred_list, weights=None):
    """
    Averages multiple prediction arrays into one ensemble prediction.
    
    Parameters:
    - pred_list: list of np.arrays, each containing model predictions
    - weights: list of floats, weights for each model (must sum to 1). If None, equal weights are used.
    
    Returns:
    - np.array: ensembled predictions
    """
    if weights is None:
        weights = [1 / len(pred_list)] * len(pred_list)
    else:
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.")
    
    pred_list = [np.array(p) for p in pred_list]
    ensemble = np.zeros_like(pred_list[0], dtype=float)
    for p, w in zip(pred_list, weights):
        ensemble += w * p
    return ensemble