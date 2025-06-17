import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import logging
import sys
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the configuration file

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Scikit-learn ---
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV


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


# --- Main Modeling Script ---
def main():
    logging.info(f"--- Starting Script: 4_modeling.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)

    X_train = load_data(config.TRAIN_X_PATH, "processed training features")
    y_train = load_data(config.TRAIN_Y_PATH, "processed training target")
    X_test = load_data(config.TEST_X_PATH, "processed test features")
    y_test = load_data(config.Y_TEST_PATH, "processed test target")

    y_train_ravel = y_train.to_numpy().ravel()
    y_test_ravel = y_test.to_numpy().ravel()

    all_results = []
    best_estimators = {}

    for threshold in config.CLUSTERING_THRESHOLDS_TO_TEST:
        feature_set_label = f"Clustered (Thresh={threshold})" if threshold is not None else "Original Features"
        logging.info(f"\n===== PROCESSING FEATURE SET: {feature_set_label} =====")

        selected_features = get_selected_features_by_clustering(X_train, threshold, config.CLUSTERING_LINKAGE_METHOD)
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test.reindex(columns=X_train_fs.columns, fill_value=0)

        logging.info(f"  Number of features: {len(selected_features)}")

        for model_name, model_template in config.MODELS_TO_BENCHMARK.items():
            logging.info(f"  --- Benchmarking Model: {model_name} ---")
            param_grid = config.PARAM_GRIDS.get(model_name, {})
            kf_cv = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE)

            grid_search = GridSearchCV(
                estimator=model_template,
                param_grid=param_grid,
                scoring=config.GRIDSEARCH_SCORING_METRIC,
                cv=kf_cv,
                n_jobs=-1,
                error_score='raise'
            )

            try:
                start_time = time.time()
                grid_search.fit(X_train_fs, y_train_ravel)
                tuning_time = time.time() - start_time
                best_estimator = grid_search.best_estimator_

                combo_key = f"{model_name}_{feature_set_label}"
                best_estimators[combo_key] = best_estimator

                result_row = {
                    "Model": model_name, "Feature Set Name": feature_set_label,
                    "Number of Features": len(selected_features), "Threshold Value": threshold,
                    "Best Params": str(grid_search.best_params_), "GridSearch Time (s)": tuning_time
                }

                cv_results = cross_validate(best_estimator, X_train_fs, y_train_ravel, cv=kf_cv,
                                            scoring=config.METRICS_TO_EVALUATE, n_jobs=-1)
                for metric in config.METRICS_TO_EVALUATE:
                    result_row[f"Mean CV {metric.replace('_', ' ').title()}"] = np.mean(cv_results[f'test_{metric}'])

                y_pred_test = best_estimator.predict(X_test_fs)
                for metric in config.METRICS_TO_EVALUATE:
                    score = f1_score(y_test_ravel, y_pred_test, average=metric.split('_')[-1],
                                     zero_division=0) if 'f1' in metric else \
                        precision_score(y_test_ravel, y_pred_test, average=metric.split('_')[-1],
                                        zero_division=0) if 'precision' in metric else \
                            recall_score(y_test_ravel, y_pred_test, average=metric.split('_')[-1],
                                         zero_division=0) if 'recall' in metric else \
                                accuracy_score(y_test_ravel, y_pred_test)
                    result_row[f"Test {metric.replace('_', ' ').title()}"] = score

                all_results.append(result_row)

            except Exception as e:
                logging.error(f"    ERROR running {model_name} for {feature_set_label}: {e}")
                continue

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(config.DETAILED_RESULTS_CSV, index=False, float_format='%.6f')
    joblib.dump(best_estimators, config.BEST_ESTIMATORS_PATH)

    logging.info(f"\nComprehensive performance results saved to: {config.DETAILED_RESULTS_CSV}")
    logging.info(f"Saved dictionary of best estimators to: {config.BEST_ESTIMATORS_PATH}")

    # ... (Plotting and detailed report logic would go here, using config variables)

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()