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
from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance


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
    """Recreates the clustered feature set for a given threshold."""
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()

    feature_names = original_df.columns.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_df = associations(original_df, nom_nom_assoc='cramer', compute_only=True)['corr'].fillna(0)

    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')

    selected_features = []
    for i in range(1, np.max(cluster_labels) + 1):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels) if label == i]
        if not cluster_indices: continue
        if len(cluster_indices) == 1:
            selected_features.append(feature_names[cluster_indices[0]])
        else:
            sum_abs_assoc = np.abs(assoc_df.iloc[cluster_indices, cluster_indices].values).sum(axis=1)
            representative_index = np.argmax(sum_abs_assoc)
            selected_features.append(feature_names[cluster_indices[representative_index]])

    return sorted(list(set(selected_features)))


# --- Main Analysis Script ---
def main():
    logging.info(f"--- Starting Script: 6_deltaAccuracy.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    results_dir = os.path.join(config.BASE_RESULTS_DIR, "importance_reports")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load Data and Models
    logging.info("Step 1: Loading data and model results...")
    X_train = load_data(config.TRAIN_X_PATH, "processed training features")
    X_test = load_data(config.TEST_X_PATH, "processed test features")
    y_test = load_data(config.Y_TEST_PATH, "processed test target")
    best_estimators = load_data(config.BEST_ESTIMATORS_PATH, "best estimators dictionary")

    try:
        performance_df = pd.read_csv(config.DETAILED_RESULTS_CSV)
    except FileNotFoundError:
        logging.error(f"FATAL: Performance results file not found at '{config.DETAILED_RESULTS_CSV}'.")
        sys.exit(1)

    # 2. Identify High-Performing Models
    logging.info(f"\nStep 2: Identifying models with Test F1 Weighted > {config.PERFORMANCE_THRESHOLD_FOR_PLOT}...")
    high_performers = performance_df[performance_df['Test F1 Weighted'] > config.PERFORMANCE_THRESHOLD_FOR_PLOT]
    if high_performers.empty:
        logging.warning("No high-performing models found to analyze. Exiting.")
        return

    # 3. Calculate Permutation Importance
    all_importances = []
    y_test_ravel = y_test.to_numpy().ravel()

    for _, row in high_performers.iterrows():
        model_name = row['Model']
        feature_set_name = row['Feature Set Name']
        combo_key = f"{model_name}_{feature_set_name}"

        logging.info(f"\n===== Analyzing Permutation Importance for: {combo_key} =====")
        estimator = best_estimators.get(combo_key)
        if estimator is None:
            logging.warning(f"  Estimator for {combo_key} not found. Skipping.")
            continue

        selected_features = get_selected_features_by_clustering(X_train, row['Threshold Value'],
                                                                config.CLUSTERING_LINKAGE_METHOD)
        X_test_fs = X_test.reindex(columns=selected_features, fill_value=0)

        scorer = make_scorer(f1_score, average='weighted', zero_division=0)

        perm_result = permutation_importance(
            estimator, X_test_fs, y_test_ravel, scoring=scorer,
            n_repeats=config.N_PERMUTATION_REPEATS, random_state=config.RANDOM_STATE, n_jobs=-1
        )

        for i, feature in enumerate(X_test_fs.columns):
            p_value = (np.sum(perm_result.importances[i] <= 0) + 1) / (config.N_PERMUTATION_REPEATS + 1)
            all_importances.append({
                'Cluster Label': feature,
                'Model (Thresh)': f"{model_name} ({'Orig' if pd.isna(row['Threshold Value']) else row['Threshold Value']})",
                'Importance (Mean Drop)': perm_result.importances_mean[i],
                'p-value': p_value
            })

    # 4. Visualize Results
    if not all_importances:
        logging.error("No importance results were generated.")
        return

    importances_df = pd.DataFrame(all_importances)
    significant_df = importances_df[importances_df['p-value'] < config.P_VALUE_THRESHOLD].copy()

    if significant_df.empty:
        logging.warning(f"No statistically significant features found with p-value < {config.P_VALUE_THRESHOLD}.")
        return

    max_importance_order = significant_df.groupby('Cluster Label')['Importance (Mean Drop)'].max().sort_values(
        ascending=False).index

    plt.style.use(config.VISUALIZATION['plot_style'])
    plt.figure(figsize=(16, max(8, len(max_importance_order) * 0.5)))
    sns.barplot(x='Importance (Mean Drop)', y='Cluster Label', hue='Model (Thresh)',
                data=significant_df, palette=config.VISUALIZATION['main_palette'], order=max_importance_order)

    plt.title(f'Statistically Significant Feature Clusters (p < {config.P_VALUE_THRESHOLD})', fontsize=16)
    plt.xlabel(f"Mean Drop in Test F1 Weighted Score", fontsize=12)
    plt.ylabel("Feature Cluster Representative", fontsize=12)
    plt.legend(title='Model (Threshold)')
    plt.tight_layout()

    plot_save_path = os.path.join(results_dir, "significant_cluster_importances.png")
    plt.savefig(plot_save_path, bbox_inches='tight')
    logging.info(f"\nFinal importance plot saved to: {plot_save_path}")
    plt.show()

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()