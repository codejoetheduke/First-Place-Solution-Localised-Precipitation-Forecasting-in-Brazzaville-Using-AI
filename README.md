# 🌧️ Localised Precipitation Forecasting in Brazzaville Using AI – Winning Solution
<img width="1024" height="1024" alt="ChatGPT Image Aug 3, 2025, 08_28_43 PM" src="https://github.com/user-attachments/assets/ac3142c8-d498-4e9b-96b1-f012bff083be" />


**How I won the Localised Precipitation Forecasting Challenge!**

This repository contains my full ML pipeline for tackling the Localised Precipitation Forecasting competition. It’s clean, modular, and fun to explore — perfect for anyone curious about how to build a winning machine learning solution.

---

## 🏆 About the Challenge

The goal was simple (well… not really):

> **Predict localised precipitation values using historical weather data.**

The dataset included meteorological features like relative humidity, pressure, and dew point temperature, collected over time. The trick? **Make accurate predictions for unseen days** while keeping the model generalizable.

I didn’t just compete. **I won.** 🎉

---

## 🧠 The Approach

My solution combined **feature engineering**, **two strong models**, and **stacking** for maximum performance:

* **Feature Engineering**: Added lags, leads, and first-order differences for key weather features.
* **Models**:

  * **CatBoost** (plain + group-wise CV)
  * **LightGBM** (plain + group-wise CV)
* **Stacking**: Blended the two models using a Ridge meta-model for final predictions.

This hybrid strategy gave me a well-balanced, robust solution.

---

## 📂 Project Structure

```
.
├── dataloader.py         # Handles data loading
├── preprocessing.py      # Feature engineering (lags, leads, diffs)
├── models.py             # CatBoostCV, LightGBMCV, and StackingRegressor
├── utils.py              # Helper functions (like ensemble predictions)
├── config.py             # Configurations for paths, features, CV splits
├── train.py              # Full training pipeline (builds & saves models)
├── inference.py          # Run inference with saved models
├── main.py               # CLI entry point for training/inference
└── README.md             # You are here
```

---

## ⚙️ How to Use

### 1. **Train the Models**

```bash
python main.py --mode train
```

This will:

* Load the training & test data
* Engineer features
* Train **CatBoost** and **LightGBM** with cross-validation
* Save all artifacts (`.pkl` models & feature engineer) in `artifacts/`

---

### 2. **Run Inference**

```bash
python main.py --mode infer --data path/to/new_data.csv --output predictions.csv
```

This will:

* Load the saved preprocessing & models
* Predict on the new dataset
* Save the predictions to `predictions.csv`

---

## 🔍 Key Features

* **Lag & Lead Features**: Capture temporal dependencies.
* **First-order Differences**: Highlight short-term changes in key weather variables.
* **Ensemble Predictions**: Blends different CV strategies for better generalization.
* **Stacking**: Uses a meta-learner (Ridge) to combine predictions for the final output.

---

## 🛠️ Requirements

* Python 3.8+
* Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ✨ Results

* **Leaderboard**: 🥇 **1st Place Winner**
* **Cross-Validation RMSE**: Consistently low across folds.
* **Key to success**: Smart feature engineering + strong ensembling.

---

## 📸 Fun

![Raindrops](https://github.com/user-attachments/assets/bf1f8b25-e19c-4af6-bb64-e0b2d565ca46)


*Because every good precipitation model deserves a nice rainy photo.*

---

## 👩‍💻 Author

**\[CodeJoe]** – Data enthusiast.
