import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import KFold, cross_validate, GridSearchCV # Added GridSearchCV

# Import Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
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
DATA_DIR = 'processed_ml_data'
RESULTS_FILENAME = 'model_tuned_cv_benchmarking_results.csv' # Updated filename

# Construct the full paths
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
FULL_RESULTS_CSV_PATH = os.path.join(DATA_DIR, RESULTS_FILENAME)
BEST_MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'best_tuned_model.pkl')

RANDOM_STATE = 42

# --- Cross-Validation & Tuning Settings ---
N_SPLITS = 5 # Number of folds for K-Fold CV (used by GridSearchCV)
# Scoring metric for GridSearchCV to optimize (choose one primary metric)
# Use 'neg_mean_squared_error' if optimizing for RMSE/MSE, 'r2' for R2, etc.
GRIDSEARCH_SCORING = 'neg_mean_squared_error' # Optimize for lowest MSE
# Scoring metrics to calculate *after* finding the best estimator
# (Used in the secondary cross_validate call for full reporting)
CV_SCORING_REPORT = {
    'neg_mae': 'neg_mean_absolute_error',
    'neg_mse': 'neg_mean_squared_error',
    'r2': 'r2'
}

# Define the base models you might want to test or tune
MODELS_TO_TEST = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE), # Sensible default if not tuned
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1), # Sensible default
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE), # Sensible default
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    # "SVR (Linear)": SVR(kernel='linear'), # Often slow
    "SVR (RBF)": SVR(kernel='rbf'),
    "KNN": KNeighborsRegressor(n_neighbors=5)
}

# --- Parameter Grids for GridSearchCV ---
# Define hyperparameters to search for each model. START SMALL!
# Comment out models you don't want to tune initially to save time.
PARAM_GRIDS = {
    "Linear Regression": {}, # No hyperparameters to tune typically
    "Ridge": {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    "Lasso": {
        'alpha': [0.01, 0.1, 1.0, 10.0] # Often needs smaller alphas than Ridge
    },
    "ElasticNet": {
        'alpha': [0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.9]
    },
    "Decision Tree": {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    },
    "Random Forest": {
        'n_estimators': [100, 200], # Reduced for speed
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3]
    },
    "Gradient Boosting": { # Can be very slow to tune
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    "Hist Gradient Boosting": { # Faster tuning
        'learning_rate': [0.05, 0.1],
        'max_leaf_nodes': [31, 63], # Default is 31
        'max_depth': [None] # Often controlled by max_leaf_nodes
    },
    # "SVR (Linear)": { # Often slow
    #     'C': [0.1, 1, 10]
    # },
    "SVR (RBF)": { # Often slow
        'C': [1, 10, 100], # Wider range often needed
        'gamma': ['scale', 0.1, 1] # 'scale' is often a good default
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    }
}
# Add grids for optional models if available
if XGB_AVAILABLE:
    PARAM_GRIDS["XGBoost"] = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
if LGBM_AVAILABLE:
    PARAM_GRIDS["LightGBM"] = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63], # Key parameter for LightGBM
        'max_depth': [-1] # -1 means no limit (often controlled by num_leaves)
    }

# Filter MODELS_TO_TEST to only include those with defined grids
MODELS_TO_TUNE = {name: model for name, model in MODELS_TO_TEST.items() if name in PARAM_GRIDS}


# Define primary metric for comparing models based on CV results
# This should correspond to a key in the final results dictionary
PRIMARY_METRIC = "Mean CV RMSE" # Based on the tuned model's CV performance
HIGHER_IS_BETTER = False # RMSE/MAE/MSE: False, R2: True

# --- Helper Functions ---
def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# --- Main Script ---
print("Starting Regression Model Benchmarking with Hyperparameter Tuning (GridSearchCV)...")
warnings.filterwarnings("ignore", category=UserWarning) # Suppress some common warnings during CV/GridSearch

# 1. Load Data
# ... (Loading logic remains the same) ...
print("\nStep 1: Loading preprocessed data...")
try:
    X_train = joblib.load(TRAIN_X_PATH)
    y_train = joblib.load(TRAIN_Y_PATH)
    X_test = joblib.load(TEST_X_PATH)
    y_test = joblib.load(TEST_Y_PATH)
    print(f"  Data loaded successfully from '{DATA_DIR}' directory.")
    # ... (print shapes) ...
except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

if isinstance(y_train, (pd.Series, pd.DataFrame)):
    y_train = y_train.values.ravel()
if isinstance(y_test, (pd.Series, pd.DataFrame)):
    y_test = y_test.values.ravel()


# 2. Run GridSearchCV for Each Model
print(f"\nStep 2: Running GridSearchCV with {N_SPLITS}-Fold CV for each model...")
tuned_results_list = []
best_estimators = {} # Store the best estimator found for each model type
# Set up K-Fold strategy (used internally by GridSearchCV)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model_name, model in MODELS_TO_TUNE.items():
    print(f"  Tuning model: {model_name}...")
    param_grid = PARAM_GRIDS[model_name]
    if not param_grid: # Handle models with no parameters to tune
        print("    No parameters to tune for this model. Using default.")
        # Optionally run simple cross_validate here if needed, or skip tuning part
        # For simplicity, we'll fit the default and evaluate later if needed
        # Or just run cross_validate on the default model now
        start_time = time.time()
        try:
            cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=CV_SCORING_REPORT, n_jobs=-1)
            best_params = {}
            best_estimator = model.fit(X_train, y_train) # Fit default on full train data
            best_estimators[model_name] = best_estimator
        except Exception as e:
             print(f"    ERROR: Model {model_name} failed during default cross-validation. Error: {e}")
             tuned_results_list.append({"Model": model_name, "Best Params": "N/A", "Mean CV MAE": np.nan, "Mean CV MSE": np.nan, "Mean CV RMSE": np.nan, "Mean CV R2": np.nan, "GridSearch Time (s)": np.nan, "Error": str(e)})
             continue # Skip to next model
        tuning_time = time.time() - start_time

    else: # Model has parameters to tune
        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=GRIDSEARCH_SCORING, # Metric to optimize during grid search
            cv=kf,
            n_jobs=-1, # Use all available cores
            verbose=0 # Set to 1 or 2 for more messages from GridSearchCV
            # refit=True by default, automatically refits best model on whole train set
        )

        try:
            grid_search.fit(X_train, y_train)
            tuning_time = time.time() - start_time

            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_ # Already refit on full X_train
            best_estimators[model_name] = best_estimator

            # Now, run cross_validate on the BEST estimator to get all metrics
            print(f"    Best params found: {best_params}")
            print(f"    Running final CV on best '{model_name}' estimator...")
            cv_results = cross_validate(
                best_estimator, X_train, y_train, cv=kf, scoring=CV_SCORING_REPORT, n_jobs=-1
            )

        except Exception as e:
            print(f"    ERROR: Model {model_name} failed during GridSearchCV. Error: {e}")
            tuned_results_list.append({"Model": model_name, "Best Params": "N/A", "Mean CV MAE": np.nan, "Mean CV MSE": np.nan, "Mean CV RMSE": np.nan, "Mean CV R2": np.nan, "GridSearch Time (s)": np.nan, "Error": str(e)})
            continue # Skip to next model

    # Process results from the cross_validate call (either default or on best estimator)
    mean_mae = -np.mean(cv_results['test_neg_mae'])
    mean_mse = -np.mean(cv_results['test_neg_mse'])
    mean_r2 = np.mean(cv_results['test_r2'])
    mean_rmse = np.sqrt(mean_mse)

    # Store results
    result_row = {
        "Model": model_name,
        "Best Params": str(best_params), # Store params as string for CSV
        "Mean CV MAE": mean_mae,
        "Mean CV MSE": mean_mse,
        "Mean CV RMSE": mean_rmse,
        "Mean CV R2": mean_r2,
        "GridSearch Time (s)": tuning_time # Time for GridSearch or default CV
    }
    tuned_results_list.append(result_row)
    print(f"    Finished tuning/CV in {tuning_time:.2f} seconds. Best Estimator Mean CV R2: {mean_r2:.4f}, Mean CV RMSE: {mean_rmse:.4f}")


# 3. Summarize Tuned CV Results
print("\nStep 3: Summarizing Tuned Cross-Validation results...")
if not tuned_results_list:
    print("No models were successfully tuned or evaluated.")
    exit()

tuned_cv_results_df = pd.DataFrame(tuned_results_list)

# Ensure primary metric column exists before sorting
if PRIMARY_METRIC not in tuned_cv_results_df.columns:
     print(f"Error: Primary metric '{PRIMARY_METRIC}' not found in results columns: {tuned_cv_results_df.columns.tolist()}")
     exit()

# Sort results based on the primary CV metric
tuned_cv_results_df_sorted = tuned_cv_results_df.sort_values(
    by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER, na_position='last'
).copy()

# --- Save Tuned CV Results to CSV ---
print(f"\nStep 3.5: Saving tuned CV results summary to {FULL_RESULTS_CSV_PATH}...")
try:
    target_dir = os.path.dirname(FULL_RESULTS_CSV_PATH)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tuned_cv_results_df_sorted.to_csv(FULL_RESULTS_CSV_PATH, index=False, float_format='%.6f')
    print(f"  Successfully saved tuned CV results to {FULL_RESULTS_CSV_PATH}")
except Exception as e:
    print(f"Error: Could not save tuned CV results to CSV at {FULL_RESULTS_CSV_PATH}. Error: {e}")

print("\n--- Tuned Cross-Validation Model Comparison (Display) ---")
# Format numeric columns for better console readability
tuned_cv_results_df_display = tuned_cv_results_df_sorted.copy()
# Select only numeric columns that actually exist in the DataFrame
numeric_cols = tuned_cv_results_df_display.select_dtypes(include=np.number).columns
for col in numeric_cols:
     if col in tuned_cv_results_df_display: # Check existence
          tuned_cv_results_df_display[col] = tuned_cv_results_df_display[col].apply(
              lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
          )
# Adjust display options if needed
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)
print(tuned_cv_results_df_display.to_string(index=False))


# 4. Identify Best Tuned Model based on CV
print("\nStep 4: Identifying best tuned model based on Cross-Validation...")
# Filter out rows with errors if any
valid_tuned_results = pd.DataFrame(tuned_results_list).dropna(subset=[PRIMARY_METRIC])

best_model_name = None
best_final_estimator = None # Store the actual best estimator object

if not valid_tuned_results.empty:
     valid_tuned_results_sorted = valid_tuned_results.sort_values(
         by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER
     )
     best_model_row = valid_tuned_results_sorted.iloc[0]
     best_model_name = best_model_row['Model']
     # Retrieve the best estimator object stored earlier during tuning (Step 2)
     if best_model_name in best_estimators: # <<< Check the correct dictionary
         best_final_estimator = best_estimators[best_model_name] # <<< Retrieve from here
         print(f"\nBest model type based on {PRIMARY_METRIC} ({'higher' if HIGHER_IS_BETTER else 'lower'} is better):")
         print(f"  Model: {best_model_row['Model']}")
         print(f"  Best Params: {best_model_row['Best Params']}")
         print(f"  {PRIMARY_METRIC}: {best_model_row[PRIMARY_METRIC]:.4f}")
         print(f"  Tuning/CV Time: {best_model_row['GridSearch Time (s)']:.2f} seconds")
     else:
          print(f"Error: Best model '{best_model_name}' not found in stored estimators dictionary.")
          best_model_name = None # Reset if estimator not found

else:
     print("Could not determine the best model from tuning results (no valid results).")


# 5. Final Evaluation of Best Tuned Model on Test Set
print("\nStep 5: Final evaluation of the best tuned model on the hold-out test set...")
# Check if we successfully identified the best model name AND retrieved its estimator object
if best_model_name and best_final_estimator:
    print(f"  Using best estimator for '{best_model_name}' (already fitted on full training data by GridSearchCV or default fit)...")

    try:
        # Predict on the test set using the best_final_estimator object
        y_pred_test = best_final_estimator.predict(X_test)

        # Calculate final metrics on the test set
        final_mae = mean_absolute_error(y_test, y_pred_test)
        final_mse = mean_squared_error(y_test, y_pred_test)
        final_rmse = np.sqrt(final_mse)
        final_r2 = r2_score(y_test, y_pred_test)

        print("\n  --- Final Test Set Performance (Best Tuned Model) ---")
        print(f"  Model: {best_model_name}")
        # Display actual params used by the final estimator
        print(f"  Params Used: {best_final_estimator.get_params() if hasattr(best_final_estimator, 'get_params') else 'N/A'}")
        print(f"  Test MAE:   {final_mae:.4f}")
        print(f"  Test MSE:   {final_mse:.4f}")
        print(f"  Test RMSE:  {final_rmse:.4f}")
        print(f"  Test R2:    {final_r2:.4f}")

        print(f"\n  Saving the best fitted model object to {BEST_MODEL_SAVE_PATH}...")
        try:
            joblib.dump(best_final_estimator, BEST_MODEL_SAVE_PATH)
            print(f"    Successfully saved best model.")
        except Exception as e:
            print(f"    Error saving best model object: {e}")

    except Exception as e:
        print(f"    ERROR: Failed to evaluate the best tuned model ('{best_model_name}') on the test set. Error: {e}")

else:
    print("  Skipping final evaluation: No best tuned model identified or best estimator object not available.")


print("\nBenchmarking with hyperparameter tuning finished.")