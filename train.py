import joblib
import pandas as pd
import numpy as np
from dataloader import load_data
from preprocessing import TimeSeriesFeatureEngineer, prepare_features
from models import CatBoostCV, LightGBMCV, StackingRegressor
from utils import ensemble_predictions
from config import *
from sklearn.linear_model import Ridge
import os

def run_training():
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)
    # Load data
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # Feature engineering
    fe = TimeSeriesFeatureEngineer(date_col='DATE', target_col='Target', diff_cols=DIFF_COLUMNS, n_lags=2)
    train_processed, test_processed = fe.transform(train_df, test_df)
    joblib.dump(fe, "artifacts/fe.pkl")
    print("artifacts/fe.pkl")

    # CatBoost
    X_cb, y_cb, X_test_cb = prepare_features(train_processed, test_processed, target_col='Target', drop_cols=DROP_COLUMNS_CAT)
    cb_plain = CatBoostCV(n_splits=N_SPLITS, use_groups=False)
    cb_plain.fit(X_cb, y_cb, X_test_cb)
    cb_group = CatBoostCV(n_splits=N_SPLITS, use_groups=True)
    cb_group.fit(X_cb, y_cb, X_test_cb, groups=train_df.get('YEAR'))
    cb_preds = ensemble_predictions([cb_plain.get_test_preds(), cb_group.get_test_preds()])
    joblib.dump((cb_plain, cb_group), "artifacts/cb_model.pkl")

    # LightGBM
    X_lgb, y_lgb, X_test_lgb = prepare_features(train_processed, test_processed, target_col='Target', drop_cols=DROP_COLUMNS_LGB)
    lgb_plain = LightGBMCV(n_splits=N_SPLITS, use_groups=False)
    lgb_plain.fit(X_lgb, y_lgb, X_test_lgb)
    lgb_group = LightGBMCV(n_splits=N_SPLITS, use_groups=True)
    lgb_group.fit(X_lgb, y_lgb, X_test_lgb, groups=train_df.get('YEAR'))
    lgb_preds = ensemble_predictions([lgb_plain.get_test_preds(), lgb_group.get_test_preds()])
    joblib.dump((lgb_plain, lgb_group), "artifacts/lgb_model.pkl")

    # Stacking
    oof_list = [(cb_plain.get_oof() + cb_group.get_oof()) / 2, (lgb_plain.get_oof() + lgb_group.get_oof()) / 2]
    test_list = [cb_preds, lgb_preds]
    stacker = StackingRegressor(meta_model=Ridge(alpha=1), n_splits=N_SPLITS)
    stacker.fit(oof_list, y_lgb.values, test_list)
    joblib.dump(stacker, "artifacts/stacker.pkl")

    print("✅ Training complete. Models saved in 'artifacts/'.")

def run_inference(output_path="submission/predictions.csv"):
    # Load data
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    # Load preprocessing and models
    fe = joblib.load("artifacts/fe.pkl")
    cb_plain, cb_group = joblib.load("artifacts/cb_model.pkl")
    lgb_plain, lgb_group = joblib.load("artifacts/lgb_model.pkl")
    stacker = joblib.load("artifacts/stacker.pkl")

    # Feature engineering (reuse training logic)
    train_processed, test_processed = fe.transform(train_df, test_df)

    # Predictions
    cb_preds = ensemble_predictions([cb_plain.get_test_preds(), cb_group.get_test_preds()])
    lgb_preds = ensemble_predictions([lgb_plain.get_test_preds(), lgb_group.get_test_preds()])

    # Stacking
    # Get final stacked predictions
    stacked_preds = stacker.get_test_preds()
    
    # Save
    submission = pd.DataFrame({'ID': test_processed['ID'], 'Target': np.clip(stacked_preds, 0, stacked_preds.max())})
    submission.to_csv(output_path, index=False)
    print(f"✅ Inference complete. Predictions saved to {output_path}")

    print(submission)