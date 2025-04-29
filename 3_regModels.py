import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate
# Import Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet # ElasticNet is here
from sklearn.tree import DecisionTreeRegressor # DecisionTreeRegressor is here
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor # GradientBoostingRegressor is here
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
# Try importing xgboost and lightgbm
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
DATA_DIR = 'processed_ml_data' # Directory where processed data was saved AND where results will be saved
RESULTS_FILENAME = 'model_benchmarking_results.csv' # Just the filename for results

# Construct the full paths using os.path.join
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
FULL_RESULTS_CSV_PATH = os.path.join(DATA_DIR, RESULTS_FILENAME) # <<< Correct full path
RANDOM_STATE = 42

# --- Cross-Validation Settings ---
N_SPLITS = 3 # Number of folds for K-Fold CV
# Define scoring metrics for cross_validate.
# Use 'neg_' prefix for error metrics as CV maximizes scores.
SCORING_CV = {
    'neg_mae': 'neg_mean_absolute_error',
    'neg_mse': 'neg_mean_squared_error',
    'r2': 'r2'
}
# We will calculate RMSE from neg_mse later

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

PRIMARY_METRIC = "Mean CV RMSE" # Choose the main metric for comparison ('RMSE', 'MAE', 'R2')
HIGHER_IS_BETTER = False # Set based on PRIMARY_METRIC (True for R2, False for MAE/MSE/RMSE)

# --- Helper Functions ---
def calculate_rmse(y_true, y_pred):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Main Script ---
print("Starting Regression Model Benchmarking with Cross-Validation...")

# 1. Load Data
print("\nStep 1: Loading preprocessed data...")
try:
    X_train = joblib.load(TRAIN_X_PATH)
    y_train = joblib.load(TRAIN_Y_PATH)
    X_test = joblib.load(TEST_X_PATH)
    y_test = joblib.load(TEST_Y_PATH)
    print(f"  Data loaded successfully from '{DATA_DIR}' directory.")
    print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"    X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"    X_train type: {type(X_train)}")
except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

if isinstance(y_train, (pd.Series, pd.DataFrame)):
    y_train = y_train.values.ravel()
if isinstance(y_test, (pd.Series, pd.DataFrame)):
    y_test = y_test.values.ravel() # Also ensure y_test is 1D array for final eval


# 2. Run Cross-Validation for Each Model
print(f"\nStep 2: Running {N_SPLITS}-Fold Cross-Validation on training data...")
cv_results_list = []
# Set up K-Fold strategy
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model_name, model in MODELS_TO_TEST.items():
    print(f"  Cross-validating model: {model_name}...")
    start_time = time.time()

    try:
        # Perform cross-validation
        cv_results = cross_validate(
            estimator=model,
            X=X_train,
            y=y_train,
            cv=kf,
            scoring=SCORING_CV,
            n_jobs=-1, # Use all available cores
            return_train_score=False # Usually not needed for basic comparison
        )
        cv_time = time.time() - start_time

        # Calculate mean metrics from CV results
        mean_fit_time = np.mean(cv_results['fit_time'])
        mean_mae = -np.mean(cv_results['test_neg_mae']) # Flip sign back
        mean_mse = -np.mean(cv_results['test_neg_mse']) # Flip sign back
        mean_r2 = np.mean(cv_results['test_r2'])
        mean_rmse = np.sqrt(mean_mse) # Calculate RMSE from positive MSE

        # Store results
        result_row = {
            "Model": model_name,
            "Mean Fit Time (s)": mean_fit_time,
            "Mean CV MAE": mean_mae,
            "Mean CV MSE": mean_mse,
            "Mean CV RMSE": mean_rmse,
            "Mean CV R2": mean_r2
        }
        cv_results_list.append(result_row)
        print(f"    Finished CV in {cv_time:.2f} seconds. Mean CV R2: {mean_r2:.4f}, Mean CV RMSE: {mean_rmse:.4f}")

    except Exception as e:
        print(f"    ERROR: Model {model_name} failed during cross-validation. Error: {e}")
        cv_results_list.append({
            "Model": model_name, "Mean Fit Time (s)": np.nan, "Mean CV MAE": np.nan,
            "Mean CV MSE": np.nan, "Mean CV RMSE": np.nan, "Mean CV R2": np.nan, "Error": str(e)
        })


# 3. Summarize CV Results
print("\nStep 3: Summarizing Cross-Validation results...")
if not cv_results_list:
    print("No models were successfully cross-validated.")
    exit()

cv_results_df = pd.DataFrame(cv_results_list)

# Ensure primary metric column exists before sorting
if PRIMARY_METRIC not in cv_results_df.columns:
     print(f"Error: Primary metric '{PRIMARY_METRIC}' not found in CV results columns: {cv_results_df.columns.tolist()}")
     exit()

# Sort results based on the primary CV metric
cv_results_df_sorted = cv_results_df.sort_values(by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER, na_position='last').copy()

# --- Save CV Results to CSV ---
print(f"\nStep 3.5: Saving CV results summary to {FULL_RESULTS_CSV_PATH}...")
try:
    target_dir = os.path.dirname(FULL_RESULTS_CSV_PATH)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    cv_results_df_sorted.to_csv(FULL_RESULTS_CSV_PATH, index=False, float_format='%.6f')
    print(f"  Successfully saved CV results to {FULL_RESULTS_CSV_PATH}")
except Exception as e:
    print(f"Error: Could not save CV results to CSV at {FULL_RESULTS_CSV_PATH}. Error: {e}")

print("\n--- Cross-Validation Model Comparison (Display) ---")
# Format numeric columns for better console readability
cv_results_df_display = cv_results_df_sorted.copy()
float_cols = cv_results_df_display.select_dtypes(include=np.number).columns
for col in float_cols:
     if col in cv_results_df_display:
          cv_results_df_display[col] = cv_results_df_display[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
print(cv_results_df_display.to_string(index=False))


# 4. Identify Best Model based on CV
print("\nStep 4: Identifying best model based on Cross-Validation...")
# Filter out rows with errors if any
valid_cv_results = pd.DataFrame(cv_results_list).dropna(subset=[PRIMARY_METRIC])

best_model_name = None
if not valid_cv_results.empty:
     # Re-sort the original numeric results to find the best index
     valid_cv_results_sorted = valid_cv_results.sort_values(by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER)
     best_model_row = valid_cv_results_sorted.iloc[0]
     best_model_name = best_model_row['Model']
     print(f"\nBest model based on {PRIMARY_METRIC} ({'higher' if HIGHER_IS_BETTER else 'lower'} is better):")
     print(f"  Model: {best_model_row['Model']}")
     print(f"  {PRIMARY_METRIC}: {best_model_row[PRIMARY_METRIC]:.4f}")
     print(f"  Mean Fit Time: {best_model_row['Mean Fit Time (s)']:.2f} seconds")
else:
     print("Could not determine the best model from CV results (no valid results).")


# 5. Final Evaluation of Best Model on Test Set (Optional but Recommended)
print("\nStep 5: Final evaluation of the best model on the hold-out test set...")
if best_model_name and best_model_name in MODELS_TO_TEST:
    print(f"  Re-training best model ('{best_model_name}') on the full training set...")
    # Instantiate a new instance of the best model
    best_model_final = MODELS_TO_TEST[best_model_name]

    try:
        start_final_fit = time.time()
        best_model_final.fit(X_train, y_train)
        final_fit_time = time.time() - start_final_fit
        print(f"    Final fit time: {final_fit_time:.2f} seconds")

        # Predict on the test set
        y_pred_test = best_model_final.predict(X_test)

        # Calculate final metrics on the test set
        final_mae = mean_absolute_error(y_test, y_pred_test)
        final_mse = mean_squared_error(y_test, y_pred_test)
        final_rmse = np.sqrt(final_mse)
        final_r2 = r2_score(y_test, y_pred_test)

        print("\n  --- Final Test Set Performance ---")
        print(f"  Model: {best_model_name}")
        print(f"  Test MAE:   {final_mae:.4f}")
        print(f"  Test MSE:   {final_mse:.4f}")
        print(f"  Test RMSE:  {final_rmse:.4f}")
        print(f"  Test R2:    {final_r2:.4f}")

    except Exception as e:
        print(f"    ERROR: Failed to re-train or evaluate the best model ('{best_model_name}') on the test set. Error: {e}")

elif best_model_name:
     print(f"  Skipping final evaluation: Best model name '{best_model_name}' not found in MODELS_TO_TEST dictionary (this shouldn't happen).")
else:
    print("  Skipping final evaluation: No best model identified from cross-validation.")


print("\nBenchmarking finished.")