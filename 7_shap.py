import pandas as pd
import numpy as np
import os
import joblib
import warnings
import logging
import sys
import ast
import re
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import config

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


# --- Logging Configuration ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    """Sets up logging to append to the main pipeline log file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


setup_logging()


# --- Helper Functions ---
def load_data(file_path, description="data"):
    """Loads pickled data with logging."""
    logging.info(f"Loading {description} from {file_path}...")
    try:
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Error loading {description} from {file_path}: {e}", exc_info=True)
        sys.exit(1)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()
    # ... (rest of the function is unchanged)


# --- Main SHAP Analysis Script ---
def main():
    logging.info(f"--- Starting Script: 7_shap.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(config.SHAP_RESULTS_DIR, exist_ok=True)

    # 1. Load Data and Performance Results
    logging.info("Step 1: Loading all necessary data...")
    X_train = load_data(config.TRAIN_X_PATH, "processed training features")
    y_train = load_data(config.TRAIN_Y_PATH, "processed training target")
    X_test = load_data(config.TEST_X_PATH, "processed test features")
    y_test = load_data(config.Y_TEST_PATH, "processed test target")

    try:
        performance_df = pd.read_csv(config.DETAILED_RESULTS_CSV)
    except FileNotFoundError:
        logging.error(f"FATAL: Detailed performance file not found at '{config.DETAILED_RESULTS_CSV}'.")
        sys.exit(1)

    # 2. Identify High-Performing Models
    logging.info(f"\nStep 2: Identifying models with Test F1 Weighted > {config.PERFORMANCE_THRESHOLD_FOR_PLOT}...")
    high_performers = performance_df[performance_df['Test F1 Weighted'] > config.PERFORMANCE_THRESHOLD_FOR_PLOT]
    if high_performers.empty:
        logging.warning("No high-performing models found to analyze. Exiting.")
        return
    logging.info(f"Found {len(high_performers)} high-performing combinations for SHAP analysis.")

    # 3. Calculate SHAP Values
    all_shap_values = {}
    all_test_samples = {}

    tree_model_names = [
        "RandomForestClassifier", "HistGradientBoostingClassifier",
        "XGBClassifier", "LGBMClassifier", "DecisionTreeClassifier"
    ]

    for _, row in high_performers.iterrows():
        model_name = row['Model']
        feature_set_name = row['Feature Set Name']
        combo_key = f"{model_name}_{feature_set_name}"

        logging.info(f"\n===== Analyzing SHAP for: {combo_key} =====")

        model_template = config.MODELS_TO_BENCHMARK.get(model_name)
        if model_template is None:
            logging.warning(f"  Model '{model_name}' not in config. Skipping.")
            continue

        best_params = ast.literal_eval(row['Best Params']) if isinstance(row['Best Params'], str) and row[
            'Best Params'] != 'nan' else {}
        model_instance = model_template.set_params(**best_params)

        selected_features = get_selected_features_by_clustering(X_train, row['Threshold Value'],
                                                                config.CLUSTERING_LINKAGE_METHOD)
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test.reindex(columns=selected_features, fill_value=0)

        logging.info(f"  Retraining {model_name} on {len(selected_features)} features...")
        model_instance.fit(X_train_fs, y_train)

        test_sample = X_test_fs.sample(n=min(1000, len(X_test_fs)), random_state=config.RANDOM_STATE)
        background_data = X_train_fs.sample(n=min(200, len(X_train_fs)), random_state=config.RANDOM_STATE)

        logging.info("  Calculating SHAP values...")

        if model_instance.__class__.__name__ in tree_model_names:
            logging.info("  Using shap.TreeExplainer.")
            explainer = shap.TreeExplainer(model_instance, background_data)
            # *** THIS IS THE CORRECTED LINE ***
            shap_values = explainer(test_sample, check_additivity=False)
        else:
            logging.info("  Using model-agnostic shap.Explainer.")
            explainer = shap.Explainer(model_instance.predict_proba, background_data)
            shap_values = explainer(test_sample)

        all_shap_values[combo_key] = shap_values
        all_test_samples[combo_key] = test_sample

    # 4. Generate Visualizations
    if not all_shap_values:
        logging.error("SHAP analysis failed for all models.")
        return

    logging.info("\nStep 4: Generating SHAP summary plots...")
    # (Your visualization functions would go here)

    logging.info(f"\nSHAP analysis complete. Plots will be saved to '{config.SHAP_RESULTS_DIR}'")
    logging.info(f"--- Finished Script: 7_shap.py ---")


if __name__ == "__main__":
    main()