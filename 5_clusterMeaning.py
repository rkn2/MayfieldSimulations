import pandas as pd
import numpy as np
import os
import joblib
import warnings
import re
import logging
import sys

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


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
BASE_RESULTS_DIR = 'cluster_exploration_results'

# Input data paths
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')

# --- USER-DEFINED CLUSTERING THRESHOLD ---
USER_DEFINED_CLUSTERING_THRESHOLD = 0.5

# --- USER-DEFINED RANDOM FEATURE ---
RANDOM_FEATURE_NAME = "num_random_feature"

CLUSTERING_LINKAGE_METHOD = 'average'


# --- Helper Functions ---
def load_pickle_data(file_path, description="data"):
    logging.info(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        logging.info(f"  Successfully loaded. Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: {description} file not found at {file_path}.")
        return None
    except Exception as e:
        logging.error(f"Error loading {description} from {file_path}: {e}", exc_info=True)
        return None


def sanitize_feature_names_df(df):
    if not isinstance(df, pd.DataFrame): return df
    original_cols = df.columns.tolist()
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_cols]
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]
    df.columns = new_cols
    return df


def calculate_association_df(dataframe):
    logging.info("  Calculating association matrix for clustering...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_results = associations(dataframe, nom_nom_assoc='cramer', compute_only=True, mark_columns=False)
    association_dataframe = assoc_results['corr']
    logging.info("  Association matrix calculated.")
    return association_dataframe.fillna(0)


def get_clusters_and_members(original_df, distance_thresh, linkage_meth):
    """
    Performs hierarchical clustering and returns a dictionary where keys are
    representative features and values are lists of all member features in that cluster.
    """
    feature_names_list = original_df.columns.tolist()
    assoc_df = calculate_association_df(original_df.copy())

    logging.info(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)

    if condensed_dist_mat.shape[0] == 0:
        if len(feature_names_list) == 1:
            return {feature_names_list[0]: feature_names_list}, 1
        logging.warning("  Condensed distance matrix is empty with multiple features. No clusters formed.")
        return {}, 0

    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    logging.info(f"  Forming flat clusters with distance threshold: {distance_thresh}...")
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    num_unique_clusters = len(np.unique(cluster_labels_arr))
    logging.info(f"  Number of clusters formed: {num_unique_clusters}")

    clusters_with_members = {}

    for i in range(1, num_unique_clusters + 1):
        cluster_member_indices_list = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_member_indices_list: continue

        current_cluster_members = [feature_names_list[idx] for idx in cluster_member_indices_list]

        if len(cluster_member_indices_list) == 1:
            representative_feature_name = feature_names_list[cluster_member_indices_list[0]]
        else:
            cluster_assoc_submat = assoc_df.iloc[cluster_member_indices_list, cluster_member_indices_list]
            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            representative_feature_name = feature_names_list[cluster_member_indices_list[np.argmax(sum_abs_assoc_arr)]]

        clusters_with_members[representative_feature_name] = sorted(current_cluster_members)

    return clusters_with_members, num_unique_clusters


# --- Main Orchestration ---
def main():
    logging.info(f"--- Starting Script: 5_clusterMeaning.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    current_clustering_threshold = USER_DEFINED_CLUSTERING_THRESHOLD
    temp_random_df = pd.DataFrame(columns=[RANDOM_FEATURE_NAME])
    sanitized_random_feature_name = sanitize_feature_names_df(temp_random_df).columns[0]

    logging.info(
        f"Using sanitized random feature name for check: '{sanitized_random_feature_name}' (original: '{RANDOM_FEATURE_NAME}')")

    results_subdir = os.path.join(BASE_RESULTS_DIR, f"threshold_{current_clustering_threshold}")
    os.makedirs(results_subdir, exist_ok=True)

    logging.info(f"Starting Cluster Member Exploration for THRESHOLD = {current_clustering_threshold}")
    logging.info(
        f"Checking for random feature: '{RANDOM_FEATURE_NAME}' (will be matched as '{sanitized_random_feature_name}')")

    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")

    if X_train_orig is None:
        logging.error("Exiting due to missing X_train data.")
        return

    X_train_sanitized = sanitize_feature_names_df(pd.DataFrame(X_train_orig))

    if not isinstance(X_train_sanitized, pd.DataFrame) or X_train_sanitized.shape[1] == 0:
        logging.error("No features available in X_train to cluster. Exiting.")
        return

    if sanitized_random_feature_name not in X_train_sanitized.columns:
        logging.warning(
            f"The specified random feature '{sanitized_random_feature_name}' was NOT FOUND in the sanitized X_train columns.")
    else:
        logging.info(f"The random feature '{sanitized_random_feature_name}' IS present in the dataset.")

    logging.info(f"\nPerforming feature clustering with threshold: {current_clustering_threshold}...")
    cluster_compositions, num_clusters = get_clusters_and_members(
        X_train_sanitized, current_clustering_threshold, CLUSTERING_LINKAGE_METHOD
    )

    if not cluster_compositions:
        logging.error("No clusters were formed or an error occurred. Exiting.")
        return

    logging.info(f"\n--- Cluster Compositions (Threshold: {current_clustering_threshold}) ---")
    logging.info(f"Total number of clusters found: {num_clusters}")

    output_lines = [f"Cluster Compositions for Threshold: {current_clustering_threshold}\n"]

    found_random_in_any_cluster = False
    for i, (representative_feature, member_features) in enumerate(cluster_compositions.items(), 1):
        is_random_cluster_flag = ""
        if sanitized_random_feature_name in member_features:
            is_random_cluster_flag = f" (Contains Random Feature: {sanitized_random_feature_name})"
            found_random_in_any_cluster = True

        header = f"\nCluster {i} (Representative: {representative_feature}){is_random_cluster_flag}"
        logging.info(header)
        output_lines.append(f"{header}\n")

        members_header = f"  Member Features ({len(member_features)}):"
        logging.info(members_header)
        output_lines.append(f"{members_header}\n")

        for member in member_features:
            member_line = f"    - {member}"
            logging.info(member_line)
            output_lines.append(f"{member_line}\n")

    if not found_random_in_any_cluster and sanitized_random_feature_name in X_train_sanitized.columns:
        logging.info(
            f"\nNote: The random feature '{sanitized_random_feature_name}' was present but not found in any cluster.")
        output_lines.append(f"\nNote: Random feature '{sanitized_random_feature_name}' was not clustered.\n")

    output_file_path = os.path.join(results_subdir, f"cluster_members_thresh_{current_clustering_threshold}.txt")
    try:
        with open(output_file_path, 'w') as f:
            f.writelines(output_lines)
        logging.info(f"\nCluster compositions also saved to text file: {output_file_path}")
    except Exception as e:
        logging.error(f"\nError saving cluster compositions to file: {e}", exc_info=True)

    logging.info("\nCluster exploration finished.")
    logging.info(f"--- Finished Script: 5_clusterMeaning.py ---")


if __name__ == '__main__':
    main()