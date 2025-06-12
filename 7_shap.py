import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import shap
import logging
import sys


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


def load_data_and_model():
    """Loads all necessary data and the trained model."""
    logging.info("--- Loading Data and Model for SHAP analysis ---")
    if not all(os.path.exists(p) for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH]):
        logging.error(
            "Error: Not all required files were found. Please ensure scripts 1, 2, and 3 have been run successfully.")
        sys.exit(1)

    try:
        X_train = joblib.load(X_TRAIN_PATH)
        X_test = joblib.load(X_TEST_PATH)
        y_test = joblib.load(Y_TEST_PATH)
        model = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logging.info("  Successfully loaded all required files.")

        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            logging.warning(
                "Warning: Processed data is not in a DataFrame format. Attempting to get feature names from preprocessor.")
            try:
                feature_names = preprocessor.get_feature_names_out()
                X_train = pd.DataFrame(X_train, columns=feature_names)
                X_test = pd.DataFrame(X_test, columns=feature_names)
            except Exception as e:
                logging.error(
                    f"Error: Could not construct DataFrame from processed data. Feature names will be missing. Error: {e}",
                    exc_info=True)
                sys.exit(1)

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
        # ** FIX: Convert DataFrames to NumPy arrays to avoid feature name issues **
        explainer = shap.Explainer(model.predict_proba, background_data.values)
        shap_values = explainer(X_test_sample.values)
        logging.info("SHAP values calculated successfully.")

    except Exception as e:
        logging.error(f"Error during SHAP value calculation: {e}", exc_info=True)
        logging.error("Please ensure the 'shap' library is installed (`pip install shap`).")
        sys.exit(1)

    unique_classes = sorted(np.unique(y_test))
    class_names = [f"Class {c}" for c in unique_classes]
    logging.info(f"\nFound unique classes in test set: {unique_classes}. Analyzing all classes.")

    # 4. Create and Save Composite Bar Chart
    logging.info("\n--- Generating and Logging Composite SHAP Bar Plot Data ---")
    mean_abs_shap_per_class = [np.abs(shap_values.values[:, :, i]).mean(0) for i in range(len(unique_classes))]
    feature_importance_df = pd.DataFrame(
        data=np.array(mean_abs_shap_per_class).T,
        index=X_test_sample.columns,
        columns=class_names
    )
    feature_importance_df['Total Importance'] = feature_importance_df.sum(axis=1)
    top_features = feature_importance_df.nlargest(N_TOP_FEATURES_TO_PLOT, 'Total Importance')

    # Log the data for the plot
    logging.info(
        f"Data for 'Top {N_TOP_FEATURES_TO_PLOT} Feature Importances by Class' plot:\n{top_features.to_string()}")

    top_features.drop(columns=['Total Importance']).plot(kind='bar', figsize=(16, 9), width=0.8, stacked=False,
                                                         colormap='viridis')
    plt.title(f'Top {N_TOP_FEATURES_TO_PLOT} Feature Importances by Class', fontsize=16)
    plt.ylabel('Mean Absolute SHAP Value (Impact on prediction)', fontsize=12)
    plt.xlabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary_bar_composite.png'))
    plt.close()
    logging.info("    Saved composite bar chart to shap_summary_bar_composite.png")

    # 5. Loop to generate individual Beeswarm plots and console output
    for i, target_class_to_explain in enumerate(unique_classes):
        logging.info(f"\n============================================================")
        logging.info(f"  ANALYZING SHAP VALUES FOR CLASS: {target_class_to_explain}")
        logging.info(f"============================================================")

        class_importance_df = pd.DataFrame({
            'Feature': X_test_sample.columns,
            'Mean Absolute SHAP Value': feature_importance_df[f'Class {target_class_to_explain}']
        }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

        logging.info(
            f"Top {N_TOP_FEATURES_TO_PLOT} features influencing predictions for class {target_class_to_explain}:")
        logging.info(f"\n{class_importance_df.head(N_TOP_FEATURES_TO_PLOT).to_string(index=False)}")

        logging.info(f"\n  Generating SHAP Beeswarm Summary Plot for Class {target_class_to_explain}...")
        plt.figure()
        # ** FIX: Use the original DataFrame for plotting to get correct feature names **
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
