import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import shap
import logging
import sys
import re


# --- Logging Configuration Setup ---
def setup_logging(log_file='pipeline.log'):
    """Sets up logging to both a file and the console."""
    # Append to the file created by previous scripts
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Ensures the handler is added even if logging was configured before
    )


# Call the setup function
setup_logging()

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
MODEL_DIR = 'processed_ml_data'
RESULTS_DIR = 'shap_plots'
BEST_MODEL_FILENAME = 'best_tuned_classifier.pkl'
PREPROCESSOR_FILENAME = 'preprocessor.pkl'

# --- Paths ---
BEST_MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, PREPROCESSOR_FILENAME)
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

# --- SHAP Configuration ---
N_SHAP_SAMPLES = 1000
N_BACKGROUND_SAMPLES = 200
N_TOP_FEATURES_TO_PLOT = 15


# *** NEW ***: Helper function to ensure consistent feature naming
def sanitize_feature_names(df):
    """Sanitizes DataFrame column names to match model's expectations."""
    if not isinstance(df, pd.DataFrame):
        return df

    # This regex replaces any character that is not a letter, number, or underscore with a single underscore.
    # It's crucial for matching names from OneHotEncoder that contain spaces or special characters.
    sanitized_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns}
    df.rename(columns=sanitized_cols, inplace=True)
    return df


def load_data_and_model():
    """Loads all necessary data and the trained model."""
    logging.info("--- Loading Data and Model for SHAP analysis ---")
    if not all(os.path.exists(p) for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH]):
        logging.error(
            "Error: Not all required files were found. Please ensure scripts 1, 2, and 3 have been run successfully.")
        sys.exit(1)

    try:
        X_train_orig = joblib.load(X_TRAIN_PATH)
        X_test_orig = joblib.load(X_TEST_PATH)
        y_test = joblib.load(Y_TEST_PATH)
        model = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logging.info("  Successfully loaded all required files.")

        # This block is crucial. It reconstructs DataFrames with proper feature names.
        if not isinstance(X_train_orig, pd.DataFrame) or not isinstance(X_test_orig, pd.DataFrame):
            logging.warning(
                "Warning: Processed data is not in a DataFrame format. Attempting to reconstruct DataFrame with feature names.")
            try:
                # Use the preprocessor to get the raw feature names
                feature_names = preprocessor.get_feature_names_out()
                X_train = pd.DataFrame(X_train_orig, columns=feature_names)
                X_test = pd.DataFrame(X_test_orig, columns=feature_names)
                logging.info("  Successfully reconstructed DataFrames.")
            except Exception as e:
                logging.error(
                    f"Error: Could not construct DataFrame from processed data. Feature names will be missing. Error: {e}",
                    exc_info=True)
                sys.exit(1)
        else:
            # If data is already a DataFrame, ensure it has column names
            X_train = X_train_orig.copy()
            X_test = X_test_orig.copy()

        # *** FIXED ***: Apply the same sanitization to the reconstructed DataFrames
        logging.info("  Sanitizing feature names to match model's fit-time names...")
        X_train = sanitize_feature_names(X_train)
        X_test = sanitize_feature_names(X_test)
        logging.info("  Feature names sanitized.")

        return X_train, X_test, y_test, model, preprocessor
    except Exception as e:
        logging.error(f"An error occurred while loading files: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main function to run SHAP analysis."""
    logging.info(f"--- Starting Script: 7_shap.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. Load data
    X_train, X_test, y_test, model, preprocessor = load_data_and_model()

    logging.info(f"\nModel loaded: {type(model).__name__}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info(f"SHAP plots will be saved to '{RESULTS_DIR}/'")

    # 2. Prepare data for SHAP
    if N_SHAP_SAMPLES and N_SHAP_SAMPLES < len(X_test):
        logging.info(f"\nUsing a random sample of {N_SHAP_SAMPLES} data points from the test set for SHAP analysis.")
        X_test_sample = X_test.sample(N_SHAP_SAMPLES, random_state=42)
    else:
        logging.info("\nUsing the full test set for SHAP analysis.")
        X_test_sample = X_test

    n_background = min(N_BACKGROUND_SAMPLES, len(X_train))
    logging.info(f"Creating background data with {n_background} samples from the training set...")
    if n_background == 0:
        logging.error("Error: Training data is empty. Cannot create a background dataset for SHAP.")
        sys.exit(1)
    background_data = X_train.sample(n_background, random_state=42)

    # 3. Initialize SHAP Explainer
    logging.info("\n--- Calculating SHAP Values ---")
    try:
        # No change needed here anymore because the DataFrames now have the correct names
        explainer = shap.Explainer(model.predict_proba, background_data)
        shap_values = explainer(X_test_sample)
        logging.info("SHAP values calculated successfully.")

    except Exception as e:
        logging.error(f"Error during SHAP value calculation: {e}", exc_info=True)
        logging.error("Please ensure the 'shap' library is installed (`pip install shap`).")
        sys.exit(1)

    unique_classes = sorted(np.unique(y_test))
    class_names = [f"Class {c}" for c in unique_classes]
    logging.info(f"\nFound unique classes in test set: {unique_classes}. Analyzing all classes.")

    # 4. Create and Save Composite Bar Chart using SHAP's built-in functionality
    logging.info("\n--- Generating Composite SHAP Bar Plot ---")
    plt.figure()
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", class_names=class_names,
                      max_display=N_TOP_FEATURES_TO_PLOT, show=False)
    plt.title(f'Top {N_TOP_FEATURES_TO_PLOT} Overall Feature Importances', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary_bar_composite.png'))
    plt.close()
    logging.info("    Saved composite bar chart to shap_summary_bar_composite.png")

    # 5. Loop to generate individual Beeswarm plots
    for i, target_class_to_explain in enumerate(unique_classes):
        logging.info(f"\n============================================================")
        logging.info(f"  ANALYZING SHAP VALUES FOR CLASS: {target_class_to_explain}")
        logging.info(f"============================================================")

        logging.info(f"\n  Generating SHAP Beeswarm Summary Plot for Class {target_class_to_explain}...")
        plt.figure()
        shap.summary_plot(shap_values[:, :, i], X_test_sample, max_display=N_TOP_FEATURES_TO_PLOT, show=False)
        plt.title(f'SHAP Summary Plot (for class {target_class_to_explain})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'shap_summary_beeswarm_class_{target_class_to_explain}.png'))
        plt.close()
        logging.info(f"    Saved shap_summary_beeswarm_class_{target_class_to_explain}.png")

    logging.info("\nSHAP analysis complete. All plots saved in the 'shap_plots' directory.")
    logging.info(f"--- Finished Script: 7_shap.py ---")


if __name__ == "__main__":
    main()

