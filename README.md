# US Treasury Bond Yield Forecasting

This project models and forecasts US Treasury bond yields using macroeconomic indicators(such as GDP, CPI, PCE, Financial Stress, Consumer Sentiment. Fiscal Deficit & Trade Deficit) with machine learning and PCA.

## Features
- PCA on multi-output yield targets (yields)
- Scree plots for the PCA components
- Lagged macroeconomic features (by a month to account for the cause and effect, as the today's macro data will impact future bond yields)
- XGBoost & Random Forest with GridSearchCV (Using regression to forecast and GridSearchCV for hyperparamter tuning)
- Time-series-aware train-test split (80/20 time split to account for trends)
- Actual vs Predicted visualizations (Using metrics such as RMSE and R^2 Squared, and plotting the actual vs predicted values)


## 🗂 Files
- `yield_modeling_final.py` — The main Python script
- `final_data_with_all_yields.csv` — Input data (local only)
- `yield_predictions_comparison.csv` — Output file with actual vs predicted values
- `plots/` — Folder containing PNG plots of yield predictions

## ▶️ How to Run
```bash
pip install -r requirements.txt
python yield_modeling_final.py
