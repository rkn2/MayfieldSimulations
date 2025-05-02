import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
RESULTS_FILENAME = 'model_tuned_cv_benchmarking_results.csv'

# Paths to load necessary files
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
BEST_MODEL_PATH = os.path.join(DATA_DIR, 'best_tuned_model.pkl')
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl')
FULL_RESULTS_CSV_PATH = os.path.join(DATA_DIR, RESULTS_FILENAME)

# Output directory for plots
PLOTS_OUTPUT_DIR = 'best_model_plots'
SAVE_PLOTS = True

# Plotting Style & Feature Importance
sns.set_style("whitegrid")
PLOT_FIGSIZE = (15, 18) # Increased height to accommodate 6 subplots
SUPTITLE_FONTSIZE = 16
SUBPLOT_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9
PLOT_FEATURE_IMPORTANCE = True
N_FEATURES_TO_SHOW = 20

# Permutation Importance Settings
CALCULATE_PERMUTATION_IMPORTANCE = True
N_PERMUTATION_REPEATS = 5 # Number of times to shuffle each feature (increase for stability)
PERMUTATION_SCORING = 'neg_mean_squared_error'
RANDOM_STATE = 42

# Confusion Matrix Settings
CONFUSION_MATRIX_NORMALIZATION = None # 'true', 'pred', 'all' or None
CONFUSION_MATRIX_DISPLAY_LABELS = None # Use None for default labels, or provide a list

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
        return data
    except Exception as e:
        print(f"Error loading data files: {e}")
        exit()

def get_feature_names(preprocessor, X_test_original_type=None):
    """Gets feature names from the preprocessor. Returns None if unable."""
    try:
        if hasattr(preprocessor, 'get_feature_names_out'):
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
best_model_val = results_df.iloc[0]['Mean CV MAE']
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

# 5. Generate Diagnostic Plots (3x2 Layout)
n_instances = len(y_test)
print("\nStep 5: Generating diagnostic plots...")

fig, axes = plt.subplots(3, 2, figsize=PLOT_FIGSIZE) # 3 rows, 2 columns
#fig.suptitle(f"Diagnostic Plots for Best Model: {type(best_model).__name__}", fontsize=SUPTITLE_FONTSIZE, y=0.98) # Adjust y
fig.suptitle(f"Diagnostic Plots for Best Model: {type(best_model).__name__} (Mean CV MAE: {best_model_val:.4f})", fontsize=SUPTITLE_FONTSIZE, y=0.98) # Adjust y

# Flatten the 2D array of axes into a 1D array for easy iteration
axes = axes.flatten()

# --- Subplot 0: Predicted vs. Actual ---
print("  Plotting Predicted vs. Actual...")
ax0 = axes[0] # First subplot
ax0.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=20)
max_val = max(np.max(y_test), np.max(y_pred)) # Use np.max for safety
min_val = min(np.min(y_test), np.min(y_pred)) # Use np.min for safety
ax0.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
ax0.set_xlabel("Actual Values (y_test)", fontsize=AXIS_LABEL_FONTSIZE)
ax0.set_ylabel("Predicted Values (y_pred)", fontsize=AXIS_LABEL_FONTSIZE)
ax0.set_title(f"Predicted vs. Actual Values (n={n_instances})", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax0.legend()
ax0.grid(True)
ax0.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

# --- Subplot 1: Residuals vs. Predicted ---
print("  Plotting Residuals vs. Predicted...")
ax1 = axes[1] # Second subplot
ax1.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=20)
ax1.axhline(0, color='red', linestyle='--', lw=2, label='Zero Error')
ax1.set_xlabel("Predicted Values (y_pred)", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_ylabel("Residuals (Actual - Predicted)", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_title(f"Residuals vs. Predicted Values (n={n_instances})", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

# --- Subplot 2: Residual Distribution ---
print("  Plotting Residual Distribution...")
ax2 = axes[2] # Third subplot
sns.histplot(residuals, kde=True, ax=ax2, bins=30)
ax2.set_xlabel("Residuals (Actual - Predicted)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Frequency", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_title(f"Distribution of Residuals (n={n_instances})", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax2.grid(True)
ax2.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)

# --- Subplot 3: Feature Importance ---
print("  Processing Feature Importance...")
ax3 = axes[3] # Fourth subplot
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
        print(f"    Standard importance not found. Calculating Permutation Importance (n_repeats={N_PERMUTATION_REPEATS}, scoring='{PERMUTATION_SCORING}')...")
        try:
            X_test_perm = X_test.values if isinstance(X_test, pd.DataFrame) else X_test # Ensure X_test is numpy
            start_perm = time.time()
            perm_result = permutation_importance(
                best_model,
                X_test_perm, # Use test set for importance calculation
                y_test,
                n_repeats=N_PERMUTATION_REPEATS,
                random_state=RANDOM_STATE,
                scoring=PERMUTATION_SCORING,
                n_jobs=-1
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

            sns.barplot(x='Score', y='Feature', data=feature_importance_df, ax=ax3, palette='rocket')
            ax3.set_title(f"Top {N_FEATURES_TO_SHOW} Features by {feature_type}", fontsize=SUBPLOT_TITLE_FONTSIZE)
            ax3.set_xlabel("Importance Score", fontsize=AXIS_LABEL_FONTSIZE)
            ax3.set_ylabel("Feature", fontsize=AXIS_LABEL_FONTSIZE)
            ax3.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE)
            ax3.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE-1)
        else:
             print(f"    Warning: Mismatch between number of features ({len(feature_names)}) and importance scores ({len(importance_scores)}). Skipping importance plot.")
             ax3.set_title("Feature Importance (Error)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
             ax3.set_axis_off()
    elif PLOT_FEATURE_IMPORTANCE: # Importance was desired but couldn't be calculated
        print(f"    Note: Could not determine feature importance for model type '{type(best_model).__name__}'. Skipping plot.")
        ax3.set_title("Feature Importance (N/A)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
        ax3.set_axis_off()

# Handle cases where plotting was disabled or feature names were missing
elif not PLOT_FEATURE_IMPORTANCE:
    print("  Note: Feature importance plot disabled by configuration.")
    ax3.set_title("Feature Importance (Disabled)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
    ax3.set_axis_off()
else: # feature_names must be None
    print("  Note: Could not retrieve feature names. Skipping feature importance plot.")
    ax3.set_title("Feature Importance (No Names)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
    ax3.set_axis_off()

# --- Subplot 4: Confusion Matrix ---
print("  Plotting Confusion Matrix...")
ax4 = axes[4] # Fifth subplot

try:
    # Create the confusion matrix
    cm = confusion_matrix(y_test, np.round(y_pred), labels=np.unique(y_test), normalize=CONFUSION_MATRIX_NORMALIZATION)

    # Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CONFUSION_MATRIX_DISPLAY_LABELS or np.unique(y_test))

    # Plot the confusion matrix on the specified axis
    disp.plot(cmap=plt.cm.Blues, ax=ax4, values_format=".2f") # values_format for cleaner display

    # Customize the subplot
    ax4.set_xlabel("Predicted Label", fontsize=AXIS_LABEL_FONTSIZE)
    ax4.set_ylabel("True Label", fontsize=AXIS_LABEL_FONTSIZE)
    ax4.set_title(f"Confusion Matrix (n={n_instances})", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax4.tick_params(axis='both', labelsize=TICK_LABEL_FONTSIZE)
    ax4.grid(False) # Remove grid lines for cleaner look

except Exception as e:
    print(f"    Error: Could not create confusion matrix. Error: {e}")
    ax4.set_title("Confusion Matrix (Error)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
    ax4.set_axis_off() # Turn off axis if there's an error

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
plt.close(fig)

print("\nBest model visualization script finished.")