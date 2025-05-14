import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- Scikit-learn ---
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score, make_scorer
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Optional Imports for Models ---
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'clustering_performance_results'  # New directory for these specific results

# Input data paths
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

RANDOM_STATE = 42
N_SPLITS_CV = 5  # For cross-validation within model tuning

# --- Clustering Configuration ---
# Thresholds to test for feature clustering. None means use all original features (baseline).
CLUSTERING_THRESHOLDS_TO_TEST = [None, 0.7]  # User requested 0.4, 0.6, 0.8
CLUSTERING_LINKAGE_METHOD = 'average'

# --- Model Benchmarking Configuration (Consistent with ordinal_classification_script_v3) ---
GRIDSEARCH_SCORING_METRIC = 'f1_weighted'  # Metric for GridSearchCV to optimize
PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL = f"Mean CV {GRIDSEARCH_SCORING_METRIC.replace('_', ' ').title()}"
METRIC_FOR_FINAL_COMPARISON_PLOT = 'Test F1 Weighted'  # Changed from 'Test Accuracy'

CV_SCORING_REPORT_DICT = {  # Metrics to calculate during cross-validation
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
}

MODELS_TO_BENCHMARK = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}
if XGB_AVAILABLE:
    MODELS_TO_BENCHMARK["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False,
                                                       eval_metric='mlogloss')
if LGBM_AVAILABLE:
    MODELS_TO_BENCHMARK["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)

PARAM_GRIDS_FOR_MODELS = {
    "Logistic Regression": {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0], 'solver': ['liblinear']},
    "Random Forest": {'n_estimators': [100, 150], 'max_depth': [10, 20], 'min_samples_leaf': [1, 3]},
    "Hist Gradient Boosting": {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 50]},
}
if XGB_AVAILABLE:
    PARAM_GRIDS_FOR_MODELS["XGBoost"] = {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
if LGBM_AVAILABLE:
    PARAM_GRIDS_FOR_MODELS["LightGBM"] = {'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1],
                                          'num_leaves': [20, 31]}


# --- Helper Functions ---

def load_pickle_data(file_path, description="data"):
    """Loads data from a pickle file."""
    print(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series) or isinstance(data, np.ndarray):
            print(f"  Successfully loaded. Shape: {data.shape if hasattr(data, 'shape') else 'N/A (Series)'}")
        else:
            print(f"  Successfully loaded. Type: {type(data)}")
        return data
    except FileNotFoundError:
        print(f"Error: {description} file not found at {file_path}.")
        return None
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}")
        return None


def sanitize_feature_names_df(df):
    """Sanitizes DataFrame column names for model compatibility."""
    if not isinstance(df, pd.DataFrame):
        return df  # Return as is if not a DataFrame
    original_cols = df.columns.tolist()
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_cols]
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]  # Further sanitization

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


# --- Feature Clustering Functions (Adapted from feature_clustering_script) ---
def calculate_association_df(dataframe):
    """Calculates the association matrix using dython."""
    print("  Calculating association matrix for clustering...")
    # Ensure boolean columns are int for dython
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool':
            dataframe[col] = dataframe[col].astype(int)
    with warnings.catch_warnings():  # Suppress dython warnings if not critical
        warnings.simplefilter("ignore")
        assoc_results = associations(dataframe, nom_nom_assoc='cramer', compute_only=True, mark_columns=False,
                                     clustering=False, plot=False)
    association_dataframe = assoc_results['corr']
    print("  Association matrix calculated.")
    return association_dataframe.fillna(0)  # Fill NaNs that might arise


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth, precomputed_assoc_df=None,
                                        precomputed_linkage_matrix=None):
    """
    Performs hierarchical clustering and returns a list of selected representative features.
    Can use precomputed association and linkage matrices for efficiency.
    """
    feature_names_list = original_df.columns.tolist()

    # Use precomputed association matrix if provided, else calculate it
    if precomputed_assoc_df is not None:
        assoc_df = precomputed_assoc_df
    else:
        assoc_df = calculate_association_df(original_df.copy())

        # Use precomputed linkage matrix if provided, else calculate it
    if precomputed_linkage_matrix is not None:
        linked = precomputed_linkage_matrix
    else:
        print(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
        distance_mat = 1 - np.abs(assoc_df.values)
        np.fill_diagonal(distance_mat, 0)  # Ensure diagonal is zero
        distance_mat = (distance_mat + distance_mat.T) / 2  # Ensure symmetry
        condensed_dist_mat = squareform(distance_mat, checks=False)  # Disable checks for minor asymmetry

        # Handle cases where condensed_dist_mat might be empty (e.g., single feature)
        if condensed_dist_mat.shape[0] == 0:
            if len(feature_names_list) == 1: return feature_names_list  # Single feature, return it
            print("  Warning: Condensed distance matrix is empty with multiple features. Returning all features.")
            return feature_names_list  # Fallback: return all features
        linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)

    print(f"  Forming flat clusters with distance threshold: {distance_thresh}...")
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    num_unique_clusters = len(np.unique(cluster_labels_arr))

    selected_representatives_list = []
    for i in range(1, num_unique_clusters + 1):  # Cluster labels are 1-indexed
        cluster_member_indices_list = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_member_indices_list: continue  # Should not happen

        if len(cluster_member_indices_list) == 1:  # Single feature in cluster
            selected_representatives_list.append(feature_names_list[cluster_member_indices_list[0]])
        else:  # Multiple features in cluster, find representative
            cluster_assoc_submat = assoc_df.iloc[cluster_member_indices_list, cluster_member_indices_list]
            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc_arr)  # Index within the sub-cluster
            rep_original_idx = cluster_member_indices_list[rep_local_idx]  # Map back to original index
            selected_representatives_list.append(feature_names_list[rep_original_idx])

    return sorted(list(set(selected_representatives_list)))  # Ensure unique and sorted


# --- Model Benchmarking Function ---
def run_benchmarking_for_feature_set(x_train_fs, y_train_fs, x_test_fs, y_test_fs, feature_set_name="Feature Set"):
    """
    Runs model tuning and evaluation for a given feature set.
    Returns a dictionary with results for the best model on this feature set.
    """
    print(f"\n--- Benchmarking for: {feature_set_name} ---")
    print(f"  X_train shape: {x_train_fs.shape}, X_test shape: {x_test_fs.shape}")

    tuned_cv_results_list_fs = []
    best_estimators_fs = {}  # Store best estimator for each model type
    kf_cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model_template_fs in MODELS_TO_BENCHMARK.items():
        print(f"    Tuning model: {model_name}...")
        param_grid_fs = PARAM_GRIDS_FOR_MODELS.get(model_name, {})

        current_cv_scoring_report_fs = CV_SCORING_REPORT_DICT.copy()
        # Note: Add logic here if specific metrics (like ROC AUC) need to be removed
        # from current_cv_scoring_report_fs if model_template_fs lacks predict_proba.

        grid_search_fs = GridSearchCV(
            estimator=model_template_fs, param_grid=param_grid_fs, scoring=GRIDSEARCH_SCORING_METRIC,
            cv=kf_cv, n_jobs=-1, verbose=0, error_score='raise'  # Raise error for debugging
        )
        try:
            grid_search_fs.fit(x_train_fs, y_train_fs)  # Fit GridSearchCV
            best_estimator_from_gs_fs = grid_search_fs.best_estimator_  # Get the best estimator
            best_estimators_fs[model_name] = best_estimator_from_gs_fs  # Store it

            # Perform cross-validation on the best estimator to get all desired metrics
            cv_results_for_model_fs = cross_validate(
                best_estimator_from_gs_fs, x_train_fs, y_train_fs, cv=kf_cv,
                scoring=current_cv_scoring_report_fs, n_jobs=-1, error_score='raise'
            )

            # Store results
            result_row_fs = {"Model": model_name, "Best Params": str(grid_search_fs.best_params_), "Error": None}
            for key_metric, scorer_name in current_cv_scoring_report_fs.items():
                cv_res_key = f'test_{key_metric}'  # Key used by cross_validate
                mean_metric_val = np.mean(cv_results_for_model_fs.get(cv_res_key, [np.nan]))  # Get mean score
                result_row_fs[f"Mean CV {key_metric.replace('_', ' ').title()}"] = mean_metric_val
            tuned_cv_results_list_fs.append(result_row_fs)
        except Exception as e_gs:
            print(f"      ERROR: Model {model_name} failed during GridSearchCV or CV: {e_gs}")
            # Record error and NaN for metrics
            error_row = {"Model": model_name, "Best Params": "N/A", "Error": str(e_gs)}
            for key_metric in current_cv_scoring_report_fs.keys():
                error_row[f"Mean CV {key_metric.replace('_', ' ').title()}"] = np.nan
            tuned_cv_results_list_fs.append(error_row)
            continue  # Move to the next model

    if not tuned_cv_results_list_fs:
        print(f"    No models successfully tuned for {feature_set_name}.")
        return None  # Return None if no models were tuned

    tuned_cv_results_df_fs = pd.DataFrame(tuned_cv_results_list_fs)

    # Identify the best model based on PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL
    if PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL not in tuned_cv_results_df_fs.columns or \
            tuned_cv_results_df_fs[PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL].isnull().all():
        print(
            f"    Warning: Primary CV metric '{PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL}' not found or all NaN for {feature_set_name}. Cannot determine best model.")
        return None  # Cannot proceed if primary metric is missing

    # Sort by the primary CV metric to find the best model
    tuned_cv_results_df_fs_sorted = tuned_cv_results_df_fs.sort_values(
        by=PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL, ascending=False, na_position='last'
        # Assuming higher is better for the metric
    ).copy()

    best_model_row_fs = tuned_cv_results_df_fs_sorted.iloc[0]  # Top row is the best model
    best_model_name_fs = best_model_row_fs["Model"]

    # Check if the best model actually has a valid score for the primary metric
    if pd.isna(best_model_row_fs[PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL]):
        print(f"    Warning: Best model for {feature_set_name} has NaN primary CV metric. Skipping test evaluation.")
        return None

    best_final_estimator_fs = best_estimators_fs.get(best_model_name_fs)  # Retrieve the stored best estimator object

    if not best_final_estimator_fs:
        print(f"    Error: Best estimator object for {best_model_name_fs} not found for {feature_set_name}.")
        return None  # Estimator object is missing

    print(
        f"    Best model for {feature_set_name} (based on CV {PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL}): {best_model_name_fs}")

    # Evaluate this single best model (from CV) on the test set
    y_pred_test_fs = best_final_estimator_fs.predict(x_test_fs)
    test_accuracy_fs = accuracy_score(y_test_fs, y_pred_test_fs)
    test_f1_weighted_fs = f1_score(y_test_fs, y_pred_test_fs, average='weighted', zero_division=0)
    test_f1_macro_fs = f1_score(y_test_fs, y_pred_test_fs, average='macro', zero_division=0)

    print(f"      Test Accuracy: {test_accuracy_fs:.4f}")
    print(f"      Test F1 Weighted: {test_f1_weighted_fs:.4f}")
    print(f"      Test F1 Macro: {test_f1_macro_fs:.4f}")

    # Return a dictionary of results for this feature set
    return {
        "Feature Set Name": feature_set_name,
        "Number of Features": x_train_fs.shape[1],
        "Best CV Model": best_model_name_fs,
        "Best CV Model Params": best_model_row_fs["Best Params"],
        PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL: best_model_row_fs[PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL],
        "Test Accuracy": test_accuracy_fs,
        "Test F1 Weighted": test_f1_weighted_fs,
        "Test F1 Macro": test_f1_macro_fs
    }


# --- Main Orchestration ---
def main():
    # Suppress common warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)  # Ensure results directory exists

    # Load original data
    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")
    y_train_orig = load_pickle_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_pickle_data(TEST_X_PATH, "original X_test")
    y_test_orig = load_pickle_data(TEST_Y_PATH, "original y_test")

    # Check if all data loaded successfully
    if X_train_orig is None or y_train_orig is None or X_test_orig is None or y_test_orig is None:
        print("Exiting due to missing data. Please check paths and ensure preprocessing was successful.")
        return

    # Sanitize feature names (important for some models like LightGBM/XGBoost)
    # This should be done before any feature selection or model training.
    # Ensure data is DataFrame before sanitizing, convert if numpy array
    X_train_sanitized = sanitize_feature_names_df(
        X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names_df(
        X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else pd.DataFrame(X_test_orig))

    # Ensure X_test has same columns as X_train after sanitization (and in same order)
    # This is critical for consistent model evaluation.
    if isinstance(X_train_sanitized, pd.DataFrame) and isinstance(X_test_sanitized, pd.DataFrame):
        train_cols = X_train_sanitized.columns
        # Reindex X_test to match X_train columns, filling missing ones with 0 (or another appropriate value)
        X_test_sanitized = X_test_sanitized.reindex(columns=train_cols, fill_value=0)

    # Ravel y_train and y_test for scikit-learn model compatibility
    y_train_ravel = y_train_orig.values.ravel() if isinstance(y_train_orig,
                                                              (pd.Series, pd.DataFrame)) else y_train_orig.ravel()
    y_test_ravel = y_test_orig.values.ravel() if isinstance(y_test_orig,
                                                            (pd.Series, pd.DataFrame)) else y_test_orig.ravel()

    all_performance_results = []  # List to store results from each feature set scenario

    # Pre-calculate association and linkage for efficiency if X_train_sanitized is a DataFrame
    precomputed_association_matrix = None
    precomputed_linkage = None
    if isinstance(X_train_sanitized, pd.DataFrame) and X_train_sanitized.shape[1] > 1:  # Only if multiple features
        print("\nPre-calculating association and linkage matrix for feature clustering...")
        precomputed_association_matrix = calculate_association_df(X_train_sanitized.copy())  # Use a copy
        distance_m = 1 - np.abs(precomputed_association_matrix.values)
        np.fill_diagonal(distance_m, 0)
        distance_m = (distance_m + distance_m.T) / 2
        condensed_dist_m = squareform(distance_m, checks=False)
        if condensed_dist_m.shape[0] > 0:  # Ensure condensed matrix is not empty
            precomputed_linkage = hierarchy.linkage(condensed_dist_m, method=CLUSTERING_LINKAGE_METHOD)
            print("Association and linkage pre-calculation complete.")
        else:
            print(
                "Warning: Could not generate condensed distance matrix for linkage. Clustering steps might select all features or fail if input is unsuitable.")

    # Iterate through each clustering threshold (None means baseline with all features)
    for threshold in CLUSTERING_THRESHOLDS_TO_TEST:
        current_X_train = X_train_sanitized  # Start with sanitized full set
        current_X_test = X_test_sanitized  # Start with sanitized full set
        feature_set_label = ""

        if threshold is None:  # Baseline case: use all original (sanitized) features
            feature_set_label = "Original Features"
            print(f"\n===== PROCESSING: {feature_set_label} ({current_X_train.shape[1]} features) =====")
            # No feature selection needed here, current_X_train/test are already set
        elif isinstance(X_train_sanitized, pd.DataFrame) and X_train_sanitized.shape[
            1] > 1 and precomputed_linkage is not None:
            # Case for clustering with a specific threshold
            feature_set_label = f"Clustered (Thresh={threshold})"
            print(f"\n===== PROCESSING: {feature_set_label} =====")
            print(f"  Applying feature clustering with threshold: {threshold}")

            selected_features = get_selected_features_by_clustering(
                X_train_sanitized,  # Pass the full sanitized training set for selection
                threshold,
                CLUSTERING_LINKAGE_METHOD,
                precomputed_assoc_df=precomputed_association_matrix,  # Use precomputed
                precomputed_linkage_matrix=precomputed_linkage  # Use precomputed
            )

            if not selected_features:  # Safety check
                print(
                    f"  Warning: No features selected by clustering for threshold {threshold}. Skipping this iteration.")
                continue
            if len(selected_features) == X_train_sanitized.shape[1]:
                print(f"  Clustering with threshold {threshold} resulted in all original features being selected.")

            print(f"  Number of features selected by clustering: {len(selected_features)}")
            # Subset X_train and X_test to the selected features
            current_X_train = X_train_sanitized[selected_features]
            current_X_test = X_test_sanitized[selected_features]
        else:  # Fallback if clustering cannot be applied (e.g., not a DataFrame, or linkage failed)
            print(
                f"\nSkipping clustering for threshold {threshold} as input is not a suitable DataFrame or precomputed linkage is missing.")
            # Decide on behavior: skip, or run with all features and label appropriately
            feature_set_label = f"Clustered (Thresh={threshold}) - SKIPPED (Using Original)"
            # For this iteration, it will use all features as current_X_train/test are not modified
            # This ensures the loop doesn't break but clearly marks the scenario.

        # Run benchmarking for the current feature set (original or clustered)
        results = run_benchmarking_for_feature_set(
            current_X_train, y_train_ravel,
            current_X_test, y_test_ravel,
            feature_set_name=feature_set_label
        )
        if results:  # If benchmarking was successful for this feature set
            results['Threshold Value'] = threshold if threshold is not None else 'N/A'  # Store threshold
            all_performance_results.append(results)

    # After iterating through all thresholds and the baseline
    if not all_performance_results:
        print("\nNo performance results were generated from any feature set. Cannot create comparison chart.")
        return

    # --- Create and Save Comparison Chart ---
    performance_df = pd.DataFrame(all_performance_results)
    print("\n\n===== Overall Performance Summary =====")
    # Display key columns from the performance DataFrame
    print(performance_df[[
        "Feature Set Name", "Number of Features", "Best CV Model",
        PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL, METRIC_FOR_FINAL_COMPARISON_PLOT, "Test F1 Weighted", "Test F1 Macro"
    ]].to_string(index=False))

    # Plotting the comparison
    plt.figure(figsize=(12, 8))  # Adjusted figure size for better label display
    sns.barplot(x="Feature Set Name", y=METRIC_FOR_FINAL_COMPARISON_PLOT, data=performance_df, palette="viridis")
    plt.title(f'{METRIC_FOR_FINAL_COMPARISON_PLOT} Comparison: Original vs. Clustered Features', fontsize=16)
    plt.xlabel("Feature Set Configuration", fontsize=12)
    plt.ylabel(METRIC_FOR_FINAL_COMPARISON_PLOT, fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)  # Adjusted rotation for better readability
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    chart_save_path = os.path.join(BASE_RESULTS_DIR,
                                   f"feature_clustering_{METRIC_FOR_FINAL_COMPARISON_PLOT.replace(' ', '_').lower()}_comparison.png")
    plt.savefig(chart_save_path)
    print(f"\nComparison bar chart saved to: {chart_save_path}")
    plt.show()  # Display the plot

    # Save detailed results to CSV
    detailed_results_path = os.path.join(BASE_RESULTS_DIR, "clustering_performance_detailed_results.csv")
    performance_df.to_csv(detailed_results_path, index=False, float_format='%.6f')
    print(f"Detailed performance results saved to: {detailed_results_path}")


if __name__ == '__main__':
    main()
