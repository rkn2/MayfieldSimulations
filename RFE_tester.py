import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import joblib
import logging
import sys


# --- Logging Configuration ---
def setup_logging(log_file='rfe_plot.log'):
    """Sets up logging for this script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # 'w' for write, to create a fresh log
            logging.StreamHandler(sys.stdout)
        ]
    )


setup_logging()

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
# We load the data that was processed but BEFORE any balancing (like SMOTE) was applied.
# RFE should be performed on the original data distribution.
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')

RANDOM_STATE = 42
# The minimum number of features to consider
MIN_FEATURES_TO_SELECT = 10
# The number of folds for cross-validation within RFECV
CV_FOLDS = 5
# The scoring metric to optimize for during feature selection
SCORING_METRIC = 'accuracy'


def plot_rfe_performance():
    """
    Performs Recursive Feature Elimination with Cross-Validation (RFECV) to find the
    optimal number of features and plots the performance.
    """
    logging.info("--- Starting RFE Performance Analysis ---")

    # --- Step 1: Load Preprocessed Data ---
    logging.info("Loading preprocessed training data...")
    try:
        # Load the data that was processed by `2_dataPreprocessing.py`
        # IMPORTANT: This should be the data BEFORE any balancing (like SMOTE) was applied.
        # We need to get the original column names from the preprocessor.
        preprocessor = joblib.load(os.path.join(DATA_DIR, 'preprocessor.pkl'))
        feature_names = preprocessor.get_feature_names_out()

        # Load the processed data and convert to DataFrame
        X_train_processed = joblib.load(TRAIN_X_PATH)
        y_train = joblib.load(TRAIN_Y_PATH)

        # Ensure X_train is a DataFrame with correct columns
        if not isinstance(X_train_processed, pd.DataFrame):
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        else:
            X_train_df = X_train_processed

        logging.info(f"  Successfully loaded training data. Shape: {X_train_df.shape}")

    except FileNotFoundError as e:
        logging.error(f"Error: Could not find data files in '{DATA_DIR}'. Please run '2_dataPreprocessing.py' first.")
        logging.error(f"Details: {e}")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}", exc_info=True)
        return

    # --- Step 2: Setup RFECV ---
    logging.info(f"Setting up RFECV with {CV_FOLDS}-fold cross-validation.")

    # Use a RandomForestClassifier as the estimator for ranking features.
    # It's robust and provides feature importances.
    estimator = RandomForestClassifier(random_state=RANDOM_STATE)

    # Use Stratified K-Fold for cross-validation to maintain class distribution.
    cv_strategy = StratifiedKFold(n_splits=CV_FOLDS)

    # The RFECV selector will automatically test different numbers of features.
    selector = RFECV(
        estimator=estimator,
        step=1,  # Remove one feature at a time
        cv=cv_strategy,
        scoring=SCORING_METRIC,
        min_features_to_select=MIN_FEATURES_TO_SELECT,
        n_jobs=-1  # Use all available CPU cores
    )

    # --- Step 3: Run the Feature Selection ---
    logging.info("Fitting RFECV... This is the longest step and may take several minutes.")
    selector.fit(X_train_df, y_train)
    logging.info("RFECV fitting complete.")

    # --- Step 4: Log and Plot the Results ---
    optimal_n_features = selector.n_features_
    logging.info(f"Optimal number of features found: {optimal_n_features}")

    # The grid_scores_ attribute holds the cross-validation scores for each number of features tested.
    # Note: In newer scikit-learn versions, this is `cv_results_['mean_test_score']`
    if hasattr(selector, 'grid_scores_'):
        performance_scores = selector.grid_scores_
    else:
        performance_scores = selector.cv_results_['mean_test_score']

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.xlabel("Number of Features Selected")
    plt.ylabel(f"Cross-Validated Score ({SCORING_METRIC.capitalize()})")
    plt.title("Recursive Feature Elimination with Cross-Validation")

    # The x-axis is the number of features selected
    x_axis_data = range(MIN_FEATURES_TO_SELECT, len(performance_scores) + MIN_FEATURES_TO_SELECT)

    plt.plot(x_axis_data, performance_scores)

    # Highlight the optimal point
    plt.axvline(x=optimal_n_features, color='r', linestyle='--',
                label=f'Optimal Features: {optimal_n_features}')

    logging.info(f"Performance scores per number of features: {performance_scores}")

    plt.legend()
    plt.grid(True)

    # Save the plot
    output_plot_path = "rfe_performance_vs_features.png"
    plt.savefig(output_plot_path)
    logging.info(f"Plot saved successfully to: {output_plot_path}")

    plt.show()


if __name__ == '__main__':
    plot_rfe_performance()
