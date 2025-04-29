import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

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

# Plotting Style
sns.set_style("whitegrid")
PLOT_FIGSIZE = (12, 10) # Figure size for 2x2 layout
SUPTITLE_FONTSIZE = 16
SUBPLOT_TITLE_FONTSIZE = 12
AXIS_LABEL_FONTSIZE = 10

# Feature Importance Plot (optional)
PLOT_FEATURE_IMPORTANCE = True
N_FEATURES_TO_SHOW = 20 # Show top N features

# --- Helper Functions ---

def load_required_data(test_x_path, test_y_path, model_path, preprocessor_path, results_path):
    """Loads test data, best model, preprocessor, and results summary."""
    print("Loading required data...")
    data = {}
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
    except FileNotFoundError as e:
        print(f"Error: Could not find required file: {e}.")
        print("Ensure the preprocessing and benchmarking scripts ran successfully and saved all outputs.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

def get_feature_names(preprocessor, X_test_original_type):
    """Gets feature names from the preprocessor."""
    try:
        # Check if X_test was originally DataFrame to get original column names if needed
        # This assumes preprocessor might need original names for passthrough columns
        if hasattr(preprocessor, 'get_feature_names_out'):
             # Use get_feature_names_out if available (newer scikit-learn versions)
             # May need input_features if passthrough columns exist and X_test was originally pandas
             # For simplicity, assume processed data columns align if not pandas originally
             feature_names = preprocessor.get_feature_names_out()
             return feature_names
        else:
            print("Warning: Preprocessor does not have 'get_feature_names_out'. Feature importance plot might lack names.")
            # Attempt fallback or return None
            return None
    except Exception as e:
        print(f"Warning: Could not retrieve feature names from preprocessor. Error: {e}")
        return None


# --- Main Visualization Script ---
print("Starting Visualization for Best Performing Model...")

# 1. Load Data and Model
loaded_data = load_required_data(
    TEST_X_PATH, TEST_Y_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH, FULL_RESULTS_CSV_PATH
)
X_test = loaded_data['X_test']
y_test = loaded_data['y_test']
best_model = loaded_data['model']
preprocessor = loaded_data['preprocessor']
results_df = loaded_data['results_df']

# Confirm model type from results (optional)
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
ax1 = axes[0, 0]
ax1.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', s=20)
# Add identity line (y=x)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
ax1.set_xlabel("Actual Values (y_test)", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_ylabel("Predicted Values (y_pred)", fontsize=AXIS_LABEL_FONTSIZE)
ax1.set_title("Predicted vs. Actual Values", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax1.legend()
ax1.grid(True)

# --- Subplot 2: Residuals vs. Predicted ---
ax2 = axes[0, 1]
ax2.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', s=20)
ax2.axhline(0, color='red', linestyle='--', lw=2, label='Zero Error')
ax2.set_xlabel("Predicted Values (y_pred)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Residuals (Actual - Predicted)", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_title("Residuals vs. Predicted Values", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax2.legend()
ax2.grid(True)

# --- Subplot 3: Residual Distribution ---
ax3 = axes[1, 0]
sns.histplot(residuals, kde=True, ax=ax3, bins=30)
ax3.set_xlabel("Residuals (Actual - Predicted)", fontsize=AXIS_LABEL_FONTSIZE)
ax3.set_ylabel("Frequency", fontsize=AXIS_LABEL_FONTSIZE)
ax3.set_title("Distribution of Residuals", fontsize=SUBPLOT_TITLE_FONTSIZE)
ax3.grid(True)

# --- Subplot 4: Feature Importance (Optional) ---
ax4 = axes[1, 1]
feature_names = get_feature_names(preprocessor, X_test) # Pass original X_test type if needed

if PLOT_FEATURE_IMPORTANCE and feature_names is not None:
    importance_scores = None
    feature_type = None

    if hasattr(best_model, 'feature_importances_'):
        importance_scores = best_model.feature_importances_
        feature_type = "Importance"
    elif hasattr(best_model, 'coef_'):
        # Use absolute value for linear model coefficients
        importance_scores = np.abs(best_model.coef_)
        # Handle potential multi-output coef shape (e.g., MultiOutputRegressor)
        if importance_scores.ndim > 1:
             importance_scores = importance_scores.mean(axis=0) # Example: average coefs
        feature_type = "Coefficient Magnitude"

    if importance_scores is not None:
        if len(importance_scores) == len(feature_names):
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Score': importance_scores
            }).sort_values(by='Score', ascending=False).head(N_FEATURES_TO_SHOW)

            sns.barplot(x='Score', y='Feature', data=feature_importance_df, ax=ax4, palette='rocket')
            ax4.set_title(f"Top {N_FEATURES_TO_SHOW} Feature {feature_type}", fontsize=SUBPLOT_TITLE_FONTSIZE)
            ax4.set_xlabel(feature_type, fontsize=AXIS_LABEL_FONTSIZE)
            ax4.set_ylabel("Feature", fontsize=AXIS_LABEL_FONTSIZE)
            ax4.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE-1) # Smaller font for feature names
        else:
             print(f"Warning: Mismatch between number of features ({len(feature_names)}) and importance scores ({len(importance_scores)}). Skipping importance plot.")
             ax4.set_title("Feature Importance (Error)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
             ax4.set_axis_off()
    else:
        print(f"Note: Model type '{type(best_model).__name__}' does not provide standard feature importance/coefficients. Skipping plot.")
        ax4.set_title("Feature Importance (N/A)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
        ax4.set_axis_off() # Turn off axis if no importance
else:
     if not PLOT_FEATURE_IMPORTANCE:
          print("Note: Feature importance plot disabled by configuration.")
     else: # feature_names must be None
          print("Note: Could not retrieve feature names. Skipping feature importance plot.")
     ax4.set_title("Feature Importance (Skipped)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
     ax4.set_axis_off()


# Adjust layout
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