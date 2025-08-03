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


class CatBoostCV:
    def __init__(self, 
                 n_splits=10, 
                 random_state=42, 
                 use_groups=False,
                 model_params=None):
        """
        CatBoost cross-validation trainer with tqdm progress.

        Parameters:
        - n_splits: int, number of folds
        - random_state: int, random seed
        - use_groups: bool, whether to use GroupKFold
        - model_params: dict, CatBoost hyperparameters
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_groups = use_groups
        self.model_params = model_params if model_params else {
            'iterations': 1000,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'early_stopping_rounds': 100,
            'verbose': 0,
            'random_seed': random_state
        }
        self.models = []
        self.oof_preds = None
        self.test_preds = None
        self.rmse_list = []

    def fit(self, X, y, X_test, groups=None):
        """
        Fits the model using cross-validation and stores OOF/test predictions.

        Parameters:
        - X: pd.DataFrame, training features
        - y: pd.Series, target
        - X_test: pd.DataFrame, test features
        - groups: pd.Series or array, group labels for GroupKFold (if used)
        """
        # Choose splitter
        if self.use_groups:
            if groups is None:
                raise ValueError("Groups must be provided for GroupKFold.")
            splitter = GroupKFold(n_splits=self.n_splits)
            split_data = splitter.split(X, y, groups)
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            split_data = splitter.split(X, y)

        # Initialize arrays
        self.oof_preds = np.zeros(len(X))
        self.test_preds = np.zeros(len(X_test))

        # Progress bar
        for train_idx, val_idx in tqdm(split_data, total=self.n_splits, desc="Training Folds"):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_pool = Pool(X_tr, y_tr)
            val_pool = Pool(X_val, y_val)

            # Model
            model = CatBoostRegressor(**self.model_params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            self.models.append(model)

            # OOF predictions
            self.oof_preds[val_idx] = model.predict(X_val)
            rmse = mean_squared_error(y_val, self.oof_preds[val_idx], squared=False)
            self.rmse_list.append(rmse)

            # Test predictions
            test_pool = Pool(X_test)
            self.test_preds += model.predict(test_pool) / self.n_splits

        print(f"\nMean CV RMSE: {np.mean(self.rmse_list):.4f}")
        return self

    def get_oof(self):
        return self.oof_preds

    def get_test_preds(self):
        return self.test_preds

    def get_models(self):
        return self.models

class LightGBMCV:
    def __init__(self, 
                 n_splits=10, 
                 random_state=42, 
                 use_groups=False,
                 model_params=None,
                 num_boost_round=1000):
        """
        LightGBM cross-validation trainer with tqdm progress.
        
        Parameters:
        - n_splits: int, number of folds
        - random_state: int, random seed
        - use_groups: bool, whether to use GroupKFold
        - model_params: dict, LightGBM hyperparameters
        - num_boost_round: int, max boosting rounds
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_groups = use_groups
        self.num_boost_round = num_boost_round
        self.model_params = model_params if model_params else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 64,
            'max_depth': -1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': random_state,
            'early_stopping_rounds': 100,
        }
        self.models = []
        self.oof_preds = None
        self.test_preds = None
        self.rmse_list = []

    def fit(self, X, y, X_test, groups=None):
        """
        Fits the model using cross-validation and stores OOF/test predictions.
        """
        # Choose splitter
        if self.use_groups:
            if groups is None:
                raise ValueError("Groups must be provided for GroupKFold.")
            splitter = GroupKFold(n_splits=self.n_splits)
            split_data = splitter.split(X, y, groups)
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            split_data = splitter.split(X, y)

        # Initialize arrays
        self.oof_preds = np.zeros(len(X))
        self.test_preds = np.zeros(len(X_test))

        # Progress bar
        for train_idx, val_idx in tqdm(split_data, total=self.n_splits, desc="Training Folds"):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_set = lgb.Dataset(X_tr, label=y_tr)
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

            model = lgb.train(
                self.model_params,
                train_set,
                num_boost_round=self.num_boost_round,
                valid_sets=[train_set, val_set],
                valid_names=['train', 'valid'],
            )
            self.models.append(model)

            # OOF predictions
            self.oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = mean_squared_error(y_val, self.oof_preds[val_idx], squared=False)
            self.rmse_list.append(rmse)

            # Test predictions
            self.test_preds += model.predict(X_test, num_iteration=model.best_iteration) / self.n_splits

        print(f"\nMean CV RMSE: {np.mean(self.rmse_list):.4f} ± {np.std(self.rmse_list):.4f}")
        return self

    def get_oof(self):
        return self.oof_preds

    def get_test_preds(self):
        return self.test_preds

    def get_models(self):
        return self.models
        

class StackingRegressor:
    def __init__(self, meta_model=None, n_splits=10, random_state=42):
        """
        Stacks multiple model predictions using a meta-model.

        Parameters:
        - meta_model: sklearn-style regressor (default: Ridge regression)
        - n_splits: int, number of CV folds for stacking
        - random_state: int, reproducibility
        """
        self.meta_model = meta_model if meta_model else Ridge(alpha=1.0)
        self.n_splits = n_splits
        self.random_state = random_state
        self.oof_preds = None
        self.test_preds = None
        self.rmse_list = []
        self.models = []

    def fit(self, oof_list, y, test_list):
        """
        Fits the stacking model.

        Parameters:
        - oof_list: list of arrays (OOF predictions from base models)
        - y: Series or array, true target values
        - test_list: list of arrays (test predictions from base models)
        """
        # Combine OOF & test predictions
        X_stack = np.column_stack(oof_list)
        X_test_stack = np.column_stack(test_list)

        # KFold for meta-model training
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.oof_preds = np.zeros(len(y))
        self.test_preds = np.zeros(len(X_test_stack))

        for train_idx, val_idx in tqdm(kf.split(X_stack, y), total=self.n_splits, desc="Stacking Folds"):
            X_tr, X_val = X_stack[train_idx], X_stack[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = self._clone_model()
            model.fit(X_tr, y_tr)
            self.models.append(model)

            # OOF predictions
            self.oof_preds[val_idx] = model.predict(X_val)
            rmse = mean_squared_error(y_val, self.oof_preds[val_idx], squared=False)
            self.rmse_list.append(rmse)

            # Test predictions
            self.test_preds += model.predict(X_test_stack) / self.n_splits

        print(f"\nStacking CV RMSE: {np.mean(self.rmse_list):.4f} ± {np.std(self.rmse_list):.4f}")
        return self

    def _clone_model(self):
        """Re-initialize the meta-model for each fold."""
        return type(self.meta_model)(**self.meta_model.get_params())

    def get_oof(self):
        return self.oof_preds

    def get_test_preds(self):
        return self.test_preds