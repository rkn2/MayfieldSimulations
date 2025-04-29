import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time # Added for timing permutation importance
import warnings
from sklearn.inspection import permutation_importance # For feature importance fallback

# --- Configuration ---
DATA_DIR = 'processed_ml_data' # Directory where processed data AND best model are saved
RESULTS_FILENAME = 'model_tuned_cv_benchmarking_results.csv' # To find best model name

# Paths to load necessary files
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
BEST_MODEL_PATH = os.path.join(DATA_DIR, 'best_tuned_model.pkl') # Path to the saved best model
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl') # Needed for feature names
FULL_RESULTS_CSV_PATH = os.path.join(DATA_DIR, RESULTS_FILENAME) # To confirm best model name

# Output directory for plots
PLOTS_OUTPUT_DIR = 'best_model_plots'
SAVE_PLOTS = True

# Plotting Style & Feature Importance
sns.set_style("whitegrid")
PLOT_FIGSIZE = (15, 12) # Figure size for 2x2 layout
SUPTITLE_FONTSIZE = 16
SUBPLOT_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
PLOT_FEATURE_IMPORTANCE = True
N_FEATURES_TO_SHOW = 20 # Show top N features

# Permutation Importance Settings (used if standard importance is unavailable)
CALCULATE_PERMUTATION_IMPORTANCE = True # Enable calculation if standard importance is missing
N_PERMUTATION_REPEATS = 5 # Number of times to shuffle each feature (increase for stability, decrease for speed)
PERMUTATION_SCORING = 'neg_mean_squared_error' # Metric to use for importance drop (use neg_ for errors)
RANDOM_STATE = 42 # For permutation reproducibility


# --- Helper Functions ---

def load_required_data(test_x_path, test_y_path, model_path, preprocessor_path, results_path):
    """Loads test data, best model, preprocessor, and results summary."""
    print("Loading required data...")
    data = {}
    all_files_found = True
    files_to_check = {
        'X_test': test_x_path,
        'y_test': test_y_path,
        'model': model_path,
        'preprocessor': preprocessor_path,
        'results_df': results_path
    }

    for key, path in files_to_check.items():
        if not os.path.exists(path):
            print(f"Error: Required file not found: {path}")
            all_files_found = False

    if not all_files_found:
        print("\nPlease ensure the preprocessing and benchmarking scripts ran successfully and saved all outputs.")
        exit()

    try:
        data['X_test'] = joblib.load(test_x_path)
        data['y_test'] = joblib.load(test_y_path)
        data['model'] = joblib.load(model_path)
        data['preprocessor'] = joblib.load(preprocessor_path)
        data['results_df'] = pd.read_csv(results_path)
        print(f"  Successfully loaded test data, model from '{model_path}', preprocessor, and results.")

        # Ensure y_test is a numpy array
        if isinstance(data['y_test'], (pd.Series, pd.DataFrame)):
            data['y_test'] = data['y_test'].values.ravel()

        # Optional: Convert X_test to DataFrame if needed for some preprocessors/feature name methods
        # This part might need adjustment based on how X_test was saved (numpy vs DataFrame)
        # if isinstance(data['X_test'], np.ndarray) and 'preprocessor' in data:
        #     try:
        #         feature_names = get_feature_names(data['preprocessor'], None)
        #         if feature_names is not None and len(feature_names) == data['X_test'].shape[1]:
        #              data['X_test'] = pd.DataFrame(data['X_test'], columns=feature_names)
        #              print("  Converted loaded X_test (numpy) to DataFrame using preprocessor feature names.")
        #     except Exception as e:
        #         print(f"  Warning: Could not convert X_test numpy array to DataFrame. {e}")

        return data
    except Exception as e:
        print(f"Error loading data files: {e}")
        exit()

def get_feature_names(preprocessor, X_test_original_type=None):
    """Gets feature names from the preprocessor. Returns None if unable."""
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
             # Assumes X_test_original_type might be needed if passthrough columns exist
             # and the original data fed to the preprocessor was a DataFrame.
             # If X_test is already processed (numpy), input_features might not be needed.
             feature_names = preprocessor.get_feature_names_out()
             print(f"  Retrieved {len(feature_names)} feature names using get_feature_names_out().")
             return list(feature_names) # Ensure it's a list
        else:
            print("Warning: Preprocessor does not have 'get_feature_names_out'. Feature names might be unavailable.")
            # Add more complex fallback logic here if needed for older sklearn versions or specific transformers
            return None
    except Exception as e:
        print(f"Warning: Could not retrieve feature names from preprocessor. Error: {e}")
        return None


# --- Main Visualization Script ---
print("Starting Visualization for Best Performing Model...")
warnings.filterwarnings("ignore", category=FutureWarning) # Suppress some seaborn/matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Load Data and Model
loaded_data = load_required_data(
    TEST_X_PATH, TEST_Y_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH, FULL_RESULTS_CSV_PATH
)
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']
best_model = loaded_data['model']
preprocessor = loaded_data['preprocessor']
results_df = loaded_data['results_df']

# Confirm model type from results (optional sanity check)
best_model_name_from_csv = results_df.iloc[0]['Model'] # Assumes CSV is sorted best first
print(f"\nLoaded model identified as: {type(best_model).__name__}")
print(f"(Best model according to results CSV: '{best_model_name_from_csv}')")


# 2. Make Predictions
print("\nStep 2: Making predictions on the test set...")
try:
    y_pred = best_model.predict(X_test)
    print("  Predictions generated successfully.")
except Exception as e:
    print(f"Error: Failed to make predictions with the loaded model. Error: {e}")
    exit()

# 3. Calculate Residuals
residuals = y_test - y_pred
print(f"  Residuals calculated (Mean: {residuals.mean():.4f}, StdDev: {residuals.std():.4f})")

# 4. Create Output Directory
if SAVE_PLOTS:
    try:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"\nPlots will be saved to: {PLOTS_OUTPUT_DIR}")
    except Exception as e:
        print(f"Warning: Could not create output directory '{PLOTS_OUTPUT_DIR}'. Plots will not be saved. Error: {e}")
        SAVE_PLOTS = False

# 5. Generate Diagnostic Plots (2x2 Layout)
print("\nStep 5: Generating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE)
fig.suptitle(f"Diagnostic Plots for Best Model: {type(best_model).__name__}", fontsize=SUPTITLE_FONTSIZE, y=1.02)

# --- Subplot 1: Predicted vs. Actual ---
print("  Plotting Predicted vs. Actual...")
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=20)
max_val = max(np.max(y_test), np.max(y_pred)) # Use np.max for safety
min_val = min(np.min(y_test), np.min(y_pred)) # Use np.min for safety
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
ax1.set_xlabel("Actual Values (y_test)", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_ylabel("Predicted Values (y_pred)", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_title("Predicted vs. Actual Values", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


# --- Subplot 2: Residuals vs. Predicted ---
print("  Plotting Residuals vs. Predicted...")
ax2 = axes[0, 1]
ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=20)
ax2.axhline(0, color='red', linestyle='--', lw=2, label='Zero Error')
ax2.set_xlabel("Predicted Values (y_pred)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Residuals (Actual - Predicted)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_title("Residuals vs. Predicted Values", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


# --- Subplot 3: Residual Distribution ---
print("  Plotting Residual Distribution...")
ax3 = axes[1, 0]
sns.histplot(residuals, kde=True, ax=ax3, bins=30)
ax3.set_xlabel("Residuals (Actual - Predicted)", fontsize=AXIS_LABEL_FONTSIZE)
ax3.set_ylabel("Frequency", fontsize=AXIS_LABEL_FONTSIZE)
ax3.set_title("Distribution of Residuals", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax3.grid(True)
ax3.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)


# --- Subplot 4: Feature Importance ---
print("  Processing Feature Importance...")
ax4 = axes[1, 1]
feature_names = get_feature_names(preprocessor, X_test) # Pass original X_test type if needed

importance_calculated = False
if PLOT_FEATURE_IMPORTANCE and feature_names is not None:
    importance_scores = None
    feature_type = None

    # Try standard attributes first
    if hasattr(best_model, 'feature_importances_'):
        importance_scores = best_model.feature_importances_
        feature_type = "Importance (Built-in)"
        importance_calculated = True
        print(f"    Using built-in '{feature_type}'.")
    elif hasattr(best_model, 'coef_'):
        importance_scores = np.abs(best_model.coef_)
        if importance_scores.ndim > 1: # Handle potential multi-output coef shape
             importance_scores = importance_scores.mean(axis=0)
        feature_type = "Coefficient Magnitude"
        importance_calculated = True
        print(f"    Using '{feature_type}'.")

    # If standard attributes not found, try Permutation Importance
    if not importance_calculated and CALCULATE_PERMUTATION_IMPORTANCE:
        print(
            f"    Standard importance not found. Calculating Permutation Importance (n_repeats={N_PERMUTATION_REPEATS}, scoring='{PERMUTATION_SCORING}')...")
        try:
            if isinstance(X_test, np.ndarray):
                if feature_names is not None and len(feature_names) == X_test.shape[1]:
                    X_test = pd.DataFrame(X_test, columns=feature_names)
                    print("      Converted X_test to DataFrame with feature names for permutation importance.")
                else:
                    print(
                        "      Warning: Could not convert X_test to DataFrame due to missing feature names or shape mismatch.")

            start_perm = time.time()
            perm_result = permutation_importance(
                best_model,
                X_test,  # X_test is now a DataFrame (if conversion was successful)
                y_test,
                n_repeats=N_PERMUTATION_REPEATS,
                random_state=RANDOM_STATE,
                scoring=PERMUTATION_SCORING,
                n_jobs=-1
                # Remove feature_names argument (not supported in this sklearn version)
            )
            duration_perm = time.time() - start_perm
            importance_scores = perm_result.importances_mean
            feature_type = f"Permutation Importance ({PERMUTATION_SCORING.split('_')[-1]})"
            importance_calculated = True
            print(f"    Permutation Importance calculated in {duration_perm:.2f} seconds.")
        except Exception as e:
            print(f"    Warning: Permutation Importance calculation failed. Error: {e}")

    # Plot if importance scores were successfully obtained
    if importance_calculated and importance_scores is not None:
        if len(importance_scores) == len(feature_names):
            print(f"    Plotting top {N_FEATURES_TO_SHOW} features by {feature_type}.")
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Score': importance_scores
            }).sort_values(by='Score', ascending=False).head(N_FEATURES_TO_SHOW)

            sns.barplot(x='Score', y='Feature', data=feature_importance_df, ax=ax4)
            ax4.set_title(f"Top {N_FEATURES_TO_SHOW} Features by {feature_type}", fontsize=SUBPLOT_TITLE_FONTSIZE)
            ax4.set_xlabel("Importance Score", fontsize=AXIS_LABEL_FONTSIZE)
            ax4.set_ylabel("Feature", fontsize=AXIS_LABEL_FONTSIZE)
            ax4.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
            ax4.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE-1) # Smaller font for potentially long feature names
        else:
             print(f"    Warning: Mismatch between number of features ({len(feature_names)}) and importance scores ({len(importance_scores)}). Skipping importance plot.")
             ax4.set_title("Feature Importance (Error)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
             ax4.set_axis_off()
    elif PLOT_FEATURE_IMPORTANCE: # Importance was desired but couldn't be calculated
        print(f"    Note: Could not determine feature importance for model type '{type(best_model).__name__}'. Skipping plot.")
        ax4.set_title("Feature Importance (N/A)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
        ax4.set_axis_off()

# Handle cases where plotting was disabled or feature names were missing
elif not PLOT_FEATURE_IMPORTANCE:
    print("  Note: Feature importance plot disabled by configuration.")
    ax4.set_title("Feature Importance (Disabled)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
    ax4.set_axis_off()
else: # feature_names must be None
    print("  Note: Could not retrieve feature names. Skipping feature importance plot.")
    ax4.set_title("Feature Importance (No Names)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
    ax4.set_axis_off()


# Adjust layout
print("\nAdjusting plot layout...")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle

# 6. Save and Show Plot
if SAVE_PLOTS:
    plot_save_path = os.path.join(PLOTS_OUTPUT_DIR, f"best_model_{type(best_model).__name__}_diagnostics.png")
    try:
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        print(f"\nDiagnostic plot saved to: {plot_save_path}")
    except Exception as e:
        print(f"Error saving diagnostic plot to {plot_save_path}: {e}")

print("\nDisplaying diagnostic plot...")
plt.show()
plt.close(fig) # Close the figure after showing/saving

print("\nBest model visualization script finished.")