import pandas as pd
import numpy as np
import os
import joblib
import warnings
import re

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'cluster_exploration_results'  # Directory for these results

# Input data paths
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')

# --- USER-DEFINED CLUSTERING THRESHOLD ---
# Set the same threshold you used for the importance analysis,
# or any other threshold you want to explore.
USER_DEFINED_CLUSTERING_THRESHOLD = 0.5  # <--- CHANGE THIS VALUE AS NEEDED

# --- USER-DEFINED RANDOM FEATURE ---
# Specify the exact name of the feature you consider "random".
# Ensure this name matches a column name in your X_train_processed.pkl
# after any sanitization if applicable (though this script sanitizes before clustering).
# If the feature name might contain special characters before sanitization,
# provide the sanitized version if you know it, or the original if sanitization is robust.
# The script will use the feature names *after* sanitization for the check.
RANDOM_FEATURE_NAME = "num_random_feature"  # <--- CHANGE THIS VALUE

CLUSTERING_LINKAGE_METHOD = 'average'


# --- Helper Functions ---
def load_pickle_data(file_path, description="data"):
    print(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        print(f"  Successfully loaded. Shape: {data.shape if hasattr(data, 'shape') else 'N/A (Series)'}")
        return data
    except FileNotFoundError:
        print(f"Error: {description} file not found at {file_path}.")
        return None
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}")
        return None


def sanitize_feature_names_df(df):
    if not isinstance(df, pd.DataFrame): return df
    original_cols = df.columns.tolist()
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_cols]
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]  # Further sanitization for LightGBM etc.
    final_cols = []
    counts = {}
    for col_name in new_cols:
        if col_name in counts:
            counts[col_name] += 1
            final_cols.append(f"{col_name}_{counts[col_name]}")
        else:
            counts[col_name] = 0
            final_cols.append(col_name)
    df.columns = final_cols
    return df


def calculate_association_df(dataframe):
    print("  Calculating association matrix for clustering...")
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool': dataframe[col] = dataframe[col].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_results = associations(dataframe, nom_nom_assoc='cramer', compute_only=True, mark_columns=False,
                                     clustering=False, plot=False)
    association_dataframe = assoc_results['corr']
    print("  Association matrix calculated.")
    return association_dataframe.fillna(0)


def get_clusters_and_members(original_df, distance_thresh, linkage_meth):
    """
    Performs hierarchical clustering and returns a dictionary where keys are
    representative features and values are lists of all member features in that cluster.
    """
    feature_names_list = original_df.columns.tolist()  # These are sanitized names
    if len(feature_names_list) <= 1:
        if feature_names_list:  # If there's one feature
            return {feature_names_list[0]: feature_names_list}, 1
        else:  # No features
            return {}, 0

    assoc_df = calculate_association_df(original_df.copy())  # original_df here is already sanitized

    print(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    distance_mat = (distance_mat + distance_mat.T) / 2
    condensed_dist_mat = squareform(distance_mat, checks=False)

    if condensed_dist_mat.shape[0] == 0:
        if len(feature_names_list) == 1:
            return {feature_names_list[0]: feature_names_list}, 1
        print("  Warning: Condensed distance matrix is empty with multiple features. No clusters formed.")
        return {}, 0

    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    print(f"  Forming flat clusters with distance threshold: {distance_thresh}...")
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    num_unique_clusters = len(np.unique(cluster_labels_arr))
    print(f"  Number of clusters formed: {num_unique_clusters}")

    clusters_with_members = {}

    for i in range(1, num_unique_clusters + 1):
        cluster_member_indices_list = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]

        if not cluster_member_indices_list: continue

        current_cluster_members = [feature_names_list[idx] for idx in cluster_member_indices_list]
        representative_feature_name = ""

        if len(cluster_member_indices_list) == 1:
            representative_feature_name = feature_names_list[cluster_member_indices_list[0]]
        else:
            cluster_assoc_submat = assoc_df.iloc[cluster_member_indices_list, cluster_member_indices_list]
            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc_arr)
            rep_original_idx = cluster_member_indices_list[rep_local_idx]
            representative_feature_name = feature_names_list[rep_original_idx]

        clusters_with_members[representative_feature_name] = sorted(current_cluster_members)

    return clusters_with_members, num_unique_clusters


# --- Main Orchestration ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    current_clustering_threshold = USER_DEFINED_CLUSTERING_THRESHOLD
    # Sanitize the user-defined random feature name to match how columns will be after sanitization
    # This assumes RANDOM_FEATURE_NAME is defined based on the *original* column name.
    # If RANDOM_FEATURE_NAME is already sanitized, this step might be redundant or slightly alter it,
    # so ensure RANDOM_FEATURE_NAME is the original name if possible.
    temp_random_df = pd.DataFrame(columns=[RANDOM_FEATURE_NAME])
    sanitized_random_feature_name = sanitize_feature_names_df(temp_random_df).columns[0]
    print(
        f"Using sanitized random feature name for check: '{sanitized_random_feature_name}' (original: '{RANDOM_FEATURE_NAME}')")

    results_subdir = os.path.join(BASE_RESULTS_DIR, f"threshold_{current_clustering_threshold}")
    os.makedirs(results_subdir, exist_ok=True)

    print(f"Starting Cluster Member Exploration for THRESHOLD = {current_clustering_threshold}")
    print(
        f"Checking for random feature: '{RANDOM_FEATURE_NAME}' (will be matched as '{sanitized_random_feature_name}')")

    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")

    if X_train_orig is None:
        print("Exiting due to missing X_train data.")
        return

    # Sanitize feature names of the loaded DataFrame
    X_train_sanitized = sanitize_feature_names_df(
        X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else pd.DataFrame(X_train_orig))

    if not isinstance(X_train_sanitized, pd.DataFrame) or X_train_sanitized.shape[1] == 0:
        print("No features available in X_train to cluster. Exiting.")
        return

    # Check if the (sanitized) random feature name actually exists in the dataset
    if sanitized_random_feature_name not in X_train_sanitized.columns:
        print(
            f"Warning: The specified random feature '{sanitized_random_feature_name}' (from original '{RANDOM_FEATURE_NAME}') was NOT FOUND in the sanitized X_train columns.")
        print("Please ensure RANDOM_FEATURE_NAME is a valid column in your X_train_processed.pkl data.")
        # We can still proceed to show clusters, but the random check won't find anything.
    else:
        print(
            f"The random feature '{sanitized_random_feature_name}' IS present in the dataset. Will check for its inclusion in clusters.")

    print(f"\nPerforming feature clustering with threshold: {current_clustering_threshold}...")
    cluster_compositions, num_clusters = get_clusters_and_members(
        X_train_sanitized,  # Pass the sanitized dataframe
        current_clustering_threshold,
        CLUSTERING_LINKAGE_METHOD
    )

    if not cluster_compositions:
        print("No clusters were formed or an error occurred. Exiting.")
        return

    print(f"\n--- Cluster Compositions (Threshold: {current_clustering_threshold}) ---")
    print(f"Total number of clusters found: {num_clusters}")

    output_lines = [f"Cluster Compositions for Threshold: {current_clustering_threshold}\n",
                    f"Checking for random feature: '{RANDOM_FEATURE_NAME}' (matched as '{sanitized_random_feature_name}')\n",
                    f"Total number of clusters found: {num_clusters}\n\n"]

    cluster_id_counter = 1
    found_random_in_any_cluster = False
    for representative_feature, member_features in cluster_compositions.items():
        is_random_cluster_flag = ""
        if sanitized_random_feature_name in member_features:
            is_random_cluster_flag = f" (Contains Random Feature: {sanitized_random_feature_name})"
            found_random_in_any_cluster = True

        header = f"Cluster {cluster_id_counter} (Representative: {representative_feature}){is_random_cluster_flag}"
        print(f"\n{header}")
        output_lines.append(f"{header}\n")

        print(f"  Member Features ({len(member_features)}):")
        output_lines.append(f"  Member Features ({len(member_features)}):\n")

        for member in member_features:
            print(f"    - {member}")
            output_lines.append(f"    - {member}\n")
        cluster_id_counter += 1

    if RANDOM_FEATURE_NAME and not found_random_in_any_cluster:
        if sanitized_random_feature_name in X_train_sanitized.columns:
            print(
                f"\nNote: The random feature '{sanitized_random_feature_name}' was present in the dataset but not found in any cluster at threshold {current_clustering_threshold}.")
            output_lines.append(
                f"\nNote: The random feature '{sanitized_random_feature_name}' was present in the dataset but not found in any cluster at threshold {current_clustering_threshold}.\n")
        else:  # Already handled by the earlier warning, but good to be explicit if it reaches here.
            pass

    # Save the output to a text file
    output_file_path = os.path.join(results_subdir, f"cluster_members_thresh_{current_clustering_threshold}.txt")
    try:
        with open(output_file_path, 'w') as f:
            f.writelines(output_lines)
        print(f"\nCluster compositions saved to: {output_file_path}")
    except Exception as e:
        print(f"\nError saving cluster compositions to file: {e}")

    print("\nCluster exploration finished. Review the output file to understand the features within each cluster.")


if __name__ == '__main__':
    main()
