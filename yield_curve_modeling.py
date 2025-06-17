# US Treasury Bond Yields Forecasting Model - Fully Commented Script

import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from sklearn.decomposition import PCA  # For dimensionality reduction of target variables
def prepare_data(df, yield_targets):
    """
    Prepares dataset for modeling by:
    - Lagging macro features (1 month)
    - Selecting only relevant yield columns
    - Dropping rows with NaNs ONLY in selected target columns
    Returns: X (features), y (targets), cleaned DataFrame
    """
    macro_cols = df.columns[:9]  # First 9 columns are macro indicators
    for col in macro_cols:
        lag_col = f"{col}_lag1"
        if lag_col not in df.columns:
            df[lag_col] = df[col].shift(1)  # Add 1-month lag

    feature_cols = [f"{col}_lag1" for col in macro_cols]  # Use only lagged features
    df_subset = df[feature_cols + yield_targets].copy()
    df_subset = df_subset.dropna(subset=yield_targets)  # Drop rows only where target yields are NaN

    X = df_subset[feature_cols]
    y = df_subset[yield_targets]
    return X, y, df_subset

# Load dataset and apply preparation
df = pd.read_csv("/content/final_data_with_all_yields.csv", index_col=0, parse_dates=True)
yield_targets = ['3 Mo', '6 Mo', '1 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '30 Yr']
X, y, df_cleaned = prepare_data(df, yield_targets)

# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA on target yields to reduce output dimensionality
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)
y_pca = PCA(n_components=3)  # Reduce to 3 principal components
y_pca_transformed = y_pca.fit_transform(y_scaled)

# Time-series aware train-test split
split_idx = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train_pca, y_test_pca = y_pca_transformed[:split_idx], y_pca_transformed[split_idx:]
y_test_actual = y.iloc[split_idx:]  # Keep actual (non-scaled) test set for evaluation

# Define models and parameter grids
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

xgb_param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [3, 5],
    'estimator__learning_rate': [0.05, 0.1]
}

rf_param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__max_depth': [5, 10]
}

# Train XGBoost model with GridSearchCV
xgb_model = GridSearchCV(
    estimator=MultiOutputRegressor(XGBRegressor(objective='reg:squarederror', random_state=42)),
    param_grid=xgb_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
xgb_model.fit(X_train, y_train_pca)
y_pca_pred_xgb = xgb_model.predict(X_test)
print("\nBest XGBoost Params:", xgb_model.best_params_)

# Train Random Forest model with GridSearchCV
rf_model = GridSearchCV(
    estimator=MultiOutputRegressor(RandomForestRegressor(random_state=42)),
    param_grid=rf_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train, y_train_pca)
y_pca_pred_rf = rf_model.predict(X_test)
print("Best Random Forest Params:", rf_model.best_params_)

# Inverse transform PCA predictions back to original yield space
y_pred_xgb = y_scaler.inverse_transform(y_pca.inverse_transform(y_pca_pred_xgb))
y_pred_rf = y_scaler.inverse_transform(y_pca.inverse_transform(y_pca_pred_rf))

# Evaluation metrics
def evaluate_model(name, y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true[yield_targets], y_pred[:, :len(yield_targets)], multioutput='raw_values'))
    r2 = r2_score(y_true[yield_targets], y_pred[:, :len(yield_targets)], multioutput='raw_values')
    print(f"\n{name} Results:")
    for i, col in enumerate(yield_targets):
        print(f"  {col} | RMSE: {rmse[i]:.4f} | R²: {r2[i]:.4f}")

evaluate_model("XGBoost with PCA targets", y_test_actual, y_pred_xgb)
evaluate_model("Random Forest with PCA targets", y_test_actual, y_pred_rf)

# Scree plot: cumulative variance explained
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(y_pca.explained_variance_ratio_), marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.title('Scree Plot (PCA on Yields)')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Bar chart: individual component variance explained
plt.figure(figsize=(10, 6))
components = np.arange(1, len(y_pca.explained_variance_ratio_) + 1)
plt.bar(components, y_pca.explained_variance_ratio_, color='skyblue', edgecolor='black')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.title('Scree Plot (Explained Variance by Component)')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(components)
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
plt.show()

# Line plots: actual vs predicted yields per maturity
maturities_to_plot = ['3 Mo', '6 Mo', '1 Yr', '5 Yr', '10 Yr', '30 Yr']
for col in maturities_to_plot:
    i = yield_targets.index(col)
    plt.figure(figsize=(12, 4))
    plt.plot(y_test_actual.index, y_test_actual[col], label='Actual', color='black')
    plt.plot(y_test_actual.index, y_pred_xgb[:, i], label='XGBoost Predicted', linestyle='--', color='blue')
    plt.plot(y_test_actual.index, y_pred_rf[:, i], label='Random Forest Predicted', linestyle='--', color='red')
    plt.title(f"Actual vs Predicted Yield: {col}")
    plt.xlabel("Date")
    plt.ylabel("Yield (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Export actual vs predicted to CSV
df_compare = pd.DataFrame(index=y_test_actual.index)
for i, col in enumerate(yield_targets):
    df_compare[f"{col}_Actual"] = y_test_actual[col].values
    df_compare[f"{col}_XGB"] = y_pred_xgb[:, i]
    df_compare[f"{col}_RF"] = y_pred_rf[:, i]
df_compare.to_csv("yield_predictions_comparison.csv")
print("\n✅ Yield prediction results exported to 'yield_predictions_comparison.csv'")
