import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # ElasticNet is here
from sklearn.tree import DecisionTreeRegressor # DecisionTreeRegressor is here
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor # GradientBoostingRegressor is here
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Optional: Try importing xgboost and lightgbm
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# --- Configuration ---
RESULTS_CSV_PATH = 'model_benchmarking_results.csv'
DATA_DIR = 'processed_ml_data' # Directory where processed data was saved
TRAIN_X_PATH = f'{DATA_DIR}/X_train_processed.pkl'
TRAIN_Y_PATH = f'{DATA_DIR}/y_train.pkl'
TEST_X_PATH = f'{DATA_DIR}/X_test_processed.pkl'
TEST_Y_PATH = f'{DATA_DIR}/y_test.pkl'

RANDOM_STATE = 42

# Define models to test - use default parameters for initial benchmark
# Using a dictionary for easy naming in results
MODELS_TO_TEST = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE), # Uncomment if desired
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE), # Limit depth initially
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), # Use more estimators
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE), # Can be slow
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "SVR (Linear)": SVR(kernel='linear'), # Can be very slow
    "SVR (RBF)": SVR(kernel='rbf'),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# Add optional models if available
if XGB_AVAILABLE:
    MODELS_TO_TEST["XGBoost"] = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)
if LGBM_AVAILABLE:
    MODELS_TO_TEST["LightGBM"] = lgb.LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)

# Define metrics to calculate
# Use functions directly from sklearn.metrics
METRICS_TO_CALCULATE = {
    "MAE": mean_absolute_error,
    "MSE": mean_squared_error,
    "R2": r2_score
    # RMSE will be calculated from MSE later
}

PRIMARY_METRIC = "RMSE" # Choose the main metric for comparison ('RMSE', 'MAE', 'R2')
HIGHER_IS_BETTER = False # Set based on PRIMARY_METRIC (True for R2, False for MAE/MSE/RMSE)

# --- Helper Functions ---
def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Main Script ---
print("Starting Regression Model Benchmarking...")

# 1. Load Data
print("\nStep 1: Loading preprocessed data...")
try:
    X_train = joblib.load(TRAIN_X_PATH)
    y_train = joblib.load(TRAIN_Y_PATH)
    X_test = joblib.load(TEST_X_PATH)
    y_test = joblib.load(TEST_Y_PATH)
    print(f"  Data loaded successfully:")
    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train shape: {y_train.shape}")
    print(f"    X_test shape: {X_test.shape}")
    print(f"    y_test shape: {y_test.shape}")
    # Check if X data is DataFrame or numpy array (affects column access later if needed)
    print(f"    X_train type: {type(X_train)}")
except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}. Ensure preprocessing script ran successfully.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Convert y_train to numpy array if it's a Series/DataFrame for consistency
if isinstance(y_train, (pd.Series, pd.DataFrame)):
    y_train = y_train.values.ravel() # Use ravel() to ensure it's 1D

# 2. Run Models and Evaluate
print("\nStep 2: Training and evaluating models...")
results = []

for model_name, model in MODELS_TO_TEST.items():
    print(f"  Testing model: {model_name}...")
    start_time = time.time()

    try:
        # Fit model
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        end_time = time.time()
        train_time = end_time - start_time

        # Calculate metrics
        metrics = {}
        for metric_name, metric_func in METRICS_TO_CALCULATE.items():
            try:
                metrics[metric_name] = metric_func(y_test, y_pred)
            except Exception as e:
                print(f"    Warning: Could not calculate metric {metric_name} for {model_name}. Error: {e}")
                metrics[metric_name] = np.nan

        # Calculate RMSE separately
        try:
            metrics["RMSE"] = calculate_rmse(y_test, y_pred)
        except Exception as e:
             print(f"    Warning: Could not calculate RMSE for {model_name}. Error: {e}")
             metrics["RMSE"] = np.nan


        # Store results
        result_row = {"Model": model_name, "Fit Time (s)": train_time}
        result_row.update(metrics)
        results.append(result_row)
        print(f"    Finished in {train_time:.2f} seconds. R2: {metrics.get('R2', 'N/A'):.4f}, RMSE: {metrics.get('RMSE', 'N/A'):.4f}")

    except Exception as e:
        print(f"    ERROR: Model {model_name} failed. Error: {e}")
        # Optionally store failure information
        results.append({"Model": model_name, "Fit Time (s)": np.nan, "MAE": np.nan, "MSE": np.nan, "R2": np.nan, "RMSE": np.nan, "Error": str(e)})


# 3. Summarize Results
print("\nStep 3: Summarizing results...")
if not results:
    print("No models were successfully tested.")
    exit()

results_df = pd.DataFrame(results)

# Ensure primary metric column exists before sorting
if PRIMARY_METRIC not in results_df.columns:
     print(f"Error: Primary metric '{PRIMARY_METRIC}' not found in results columns: {results_df.columns.tolist()}")
     print("Available metrics:", [col for col in results_df.columns if col not in ["Model", "Fit Time (s)", "Error"]])
     exit()

# Sort results (using the numeric data)
results_df_sorted = results_df.sort_values(by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER, na_position='last').copy() # Use copy to avoid SettingWithCopyWarning if modifying later

# --- Save Results to CSV
print(f"\nStep 3.5: Saving results summary to {RESULTS_CSV_PATH}...")
try:
    # Ensure the directory exists if RESULTS_CSV_PATH includes one
    output_dir = os.path.dirname(RESULTS_CSV_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  Created directory: {output_dir}")

    # Save the sorted DataFrame with numeric values
    results_df_sorted.to_csv(DATA_DIR+RESULTS_CSV_PATH, index=False, float_format='%.6f') # Save with good precision
    print(f"  Successfully saved results to {RESULTS_CSV_PATH}")
except Exception as e:
    print(f"Error: Could not save results to CSV. Error: {e}")

print("\n--- Model Comparison ---")
# Format numeric columns for better readability
float_cols = results_df.select_dtypes(include=np.number).columns
results_df[float_cols] = results_df[float_cols].applymap(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
print(results_df.to_string(index=False)) # Use to_string to see all columns/rows

# 4. Identify Best Model
print("\nStep 4: Identifying best model...")
# Filter out rows with errors if any
valid_results = pd.DataFrame(results).dropna(subset=[PRIMARY_METRIC])

if not valid_results.empty:
     # Re-sort the original numeric results to find the best index
     valid_results_sorted = valid_results.sort_values(by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER)
     best_model_row = valid_results_sorted.iloc[0]
     print(f"\nBest model based on {PRIMARY_METRIC} ({'higher' if HIGHER_IS_BETTER else 'lower'} is better):")
     print(f"  Model: {best_model_row['Model']}")
     print(f"  {PRIMARY_METRIC}: {best_model_row[PRIMARY_METRIC]:.4f}")
     print(f"  Fit Time: {best_model_row['Fit Time (s)']:.2f} seconds")
else:
     print("Could not determine the best model (no valid results).")


print("\nBenchmarking finished.")