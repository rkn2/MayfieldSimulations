import pandas as pd
import numpy as np
import joblib
import os
import logging
import sys


# --- Logging Configuration ---
def setup_logging(log_file='pipeline.log'):
    """Sets up logging for this script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )


setup_logging()

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
RESULTS_DIR = 'clustering_performance_results'

# Paths to the saved model and preprocessor from your pipeline
BEST_ESTIMATORS_PATH = os.path.join(RESULTS_DIR, 'best_estimators_per_combo.pkl')


# We no longer need the preprocessor for this script
# PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl')


# --- Main Analysis Function ---
def analyze_coefficients():
    logging.info("--- Starting Coefficient Analysis for Best Model ---")

    # --- Step 1: Load the necessary files ---
    try:
        # We only need the dictionary of trained models
        best_estimators = joblib.load(BEST_ESTIMATORS_PATH)
        logging.info("Successfully loaded best estimators dictionary.")
    except FileNotFoundError as e:
        logging.error(
            f"Error: Could not find required file. Please ensure '4_clustering.py' has been run successfully and saved the estimators.")
        logging.error(f"Details: {e}")
        return

    # --- Step 2: Identify the best model and get its details ---
    # The key is constructed as "ModelName_FeatureSetName"
    best_model_key = 'Logistic Regression_Original Features'

    if best_model_key not in best_estimators:
        logging.error(f"Could not find the specified best model '{best_model_key}' in the saved dictionary.")
        return

    best_model = best_estimators[best_model_key]
    logging.info(f"Analyzing coefficients for model: {best_model_key}")

    # --- Step 3: Extract feature names and coefficients ---
    try:
        # *** FIXED: Get feature names directly from the fitted model object ***
        # This ensures we get the exact features the model was trained on (e.g., the 50 from RFE)
        if hasattr(best_model, 'feature_names_in_'):
            feature_names = best_model.feature_names_in_
        else:
            logging.error(
                "Could not retrieve feature names from the model. The model may not have been fitted on a DataFrame.")
            return

        coefficients = best_model.coef_

        # Check for shape consistency before creating the DataFrame
        if coefficients.shape[1] != len(feature_names):
            raise ValueError(
                f"Shape mismatch! Model has {coefficients.shape[1]} coefficients, but there are {len(feature_names)} feature names.")

    except Exception as e:
        logging.error(f"Could not extract feature names or coefficients. Error: {e}")
        return

    # --- Step 4: Create and display the results DataFrame ---
    # For multi-class models, coefficients are shaped (n_classes, n_features)
    # We will create a table with features as rows and class coefficients as columns
    coef_df = pd.DataFrame(coefficients.T, index=feature_names,
                           columns=[f'Coef_Class_{c}' for c in best_model.classes_])

    # Sort by the coefficient for a specific class to see the most influential features
    # For example, let's sort by Class 2 (Demolished) to see what drives that prediction
    sort_by_class = f'Coef_Class_{best_model.classes_[-1]}'
    coef_df_sorted = coef_df.sort_values(by=sort_by_class, ascending=False)

    logging.info("\n--- Logistic Regression Coefficients ---")
    # Use pandas options to display all rows
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(coef_df_sorted)

    # Save the coefficients to a CSV file for further analysis
    output_path = os.path.join(RESULTS_DIR, 'best_model_coefficients.csv')
    try:
        coef_df_sorted.to_csv(output_path)
        logging.info(f"\nSuccessfully saved coefficients to: {output_path}")
    except Exception as e:
        logging.error(f"Failed to save coefficients CSV. Error: {e}")


if __name__ == '__main__':
    analyze_coefficients()

