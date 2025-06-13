import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
import sys
import json

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Scikit-learn ---
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Import Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# --- Optional Imports ---
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
try:  # Added MORD import check
    import mord

    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False


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
        force=True  # Force re-configuration to append handlers
    )


# Call the setup function
setup_logging()

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'clustering_performance_results'
BEST_THRESHOLD_FILE = os.path.join(BASE_RESULTS_DIR, 'best_threshold.json')
TOP_MODELS_FILE = os.path.join(BASE_RESULTS_DIR, 'top_models.json')
# Paths to pre-split and preprocessed data
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

# Number of times to repeat the entire analysis on fixed splits
N_REPEATS_ANALYSIS = 3  # You can increase this for more robust variance estimation

RANDOM_STATE = 42  # Base random state for reproducibility
N_SPLITS_CV = 5
CLUSTERING_THRESHOLDS_TO_TEST = [None, 0.8, 0.9]  # Test thresholds
CLUSTERING_LINKAGE_METHOD = 'average'
GRIDSEARCH_SCORING_METRIC = 'f1_weighted'
PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL = f"Mean CV {GRIDSEARCH_SCORING_METRIC.replace('_', ' ').title()}"
METRIC_FOR_FINAL_COMPARISON_PLOT = 'Test F1 Weighted'
N_TOP_MODELS_TO_SAVE = 3

CV_SCORING_REPORT_DICT = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
}

# --- Model Definitions (Copied from 3_classModel.py) ---
NORMAL_MODELS_TO_TEST = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier()
}
if XGB_AVAILABLE:
    NORMAL_MODELS_TO_TEST["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
if LGBM_AVAILABLE:
    NORMAL_MODELS_TO_TEST["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)

NORMAL_PARAM_GRIDS = {
    "Logistic Regression": {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]},
    "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8, 10], 'min_samples_leaf': [5, 10, 15]},
    "Random Forest": {'n_estimators': [100, 150], 'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 10]},
    "Gradient Boosting": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]},
    "Hist Gradient Boosting": {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [20, 31]},
    "KNN": {'n_neighbors': [5, 7, 9], 'weights': ['uniform', 'distance']}
}
if XGB_AVAILABLE:
    NORMAL_PARAM_GRIDS["XGBoost"] = {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]}
if LGBM_AVAILABLE:
    NORMAL_PARAM_GRIDS["LightGBM"] = {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'num_leaves': [15, 25]}

ORDINAL_MODELS_TO_TEST = {}
ORDINAL_PARAM_GRIDS = {}
if MORD_AVAILABLE:
    ORDINAL_MODELS_TO_TEST = {
        "Ordinal Logistic (AT)": mord.LogisticAT(), "Ordinal Ridge": mord.OrdinalRidge(), "Ordinal LAD": mord.LAD()
    }
    ORDINAL_PARAM_GRIDS = {
        "Ordinal Logistic (AT)": {'alpha': [0.1, 1.0, 10.0]}, "Ordinal Ridge": {'alpha': [0.1, 1.0, 10.0]},
        "Ordinal LAD": {'C': [0.1, 1.0, 10.0]}
    }

# Combine Normal and Ordinal models for benchmarking as per 3_classModel's 'both' setting
MODELS_TO_BENCHMARK = NORMAL_MODELS_TO_TEST.copy()
PARAM_GRIDS_FOR_MODELS = NORMAL_PARAM_GRIDS.copy()
if MORD_AVAILABLE:
    MODELS_TO_BENCHMARK.update(ORDINAL_MODELS_TO_TEST)
    PARAM_GRIDS_FOR_MODELS.update(ORDINAL_PARAM_GRIDS)

PRIMARY_CV_METRIC = f"Mean CV {GRIDSEARCH_SCORING_METRIC.replace('_', ' ').title()}"


def load_pickle_data(file_path, description="data"):
    logging.info(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        if hasattr(data, 'shape'):
            logging.info(f"  Successfully loaded. Shape: {data.shape}")
        else:
            logging.info(f"  Successfully loaded. Type: {type(data)}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: {description} file not found at {file_path}. Exiting.")
        return None
    except Exception as e:
        logging.error(f"Error loading {description} from {file_path}: {e}", exc_info=True)
        return None


def sanitize_feature_names_df(df):
    if not isinstance(df, pd.DataFrame): return df
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    df.columns = new_cols
    return df


def calculate_association_df(dataframe):
    logging.info("  Calculating association matrix for clustering...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ensure only nominal columns are passed, Cramer's V is for nominal-nominal.
        nominal_cols = dataframe.select_dtypes(include='object').columns.tolist()
        if nominal_cols:
            assoc_results = associations(dataframe[nominal_cols], nom_nom_assoc='cramer', compute_only=True,
                                         mark_columns=False)
            association_dataframe = assoc_results['corr']
        else:  # If no nominal columns, create a correlation matrix for numerical features.
            logging.info("  No nominal columns found. Association matrix will be based on numerical features.")
            if not dataframe.empty:
                association_dataframe = dataframe.corr().abs()
                np.fill_diagonal(association_dataframe.values, 1.0)
            else:
                return pd.DataFrame()  # Return empty if no features at all.

    logging.info("  Association matrix calculated.")
    return association_dataframe.fillna(0)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth, precomputed_assoc_df=None,
                                        precomputed_linkage_matrix=None):
    feature_names_list = original_df.columns.tolist()

    if len(feature_names_list) <= 1:
        logging.info("  Only one or no features, returning all available.")
        return feature_names_list

    # Ensure association matrix is computed for current feature set if not precomputed
    if precomputed_assoc_df is None:
        assoc_df = calculate_association_df(original_df.copy())
    else:
        # Reindex the precomputed_assoc_df to match the current original_df columns
        assoc_df = precomputed_assoc_df.reindex(index=feature_names_list, columns=feature_names_list).fillna(0)

    if precomputed_linkage_matrix is not None:
        linked = precomputed_linkage_matrix
    else:
        logging.info(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
        # Fix: Ensure non-negative distances for linkage
        distance_mat = np.maximum(0, 1 - np.abs(assoc_df.values))
        np.fill_diagonal(distance_mat, 0)  # Distance to self is 0

        if distance_mat.shape[0] == 0:
            logging.warning("  Distance matrix is empty. Cannot perform clustering. Returning all features.")
            return feature_names_list

        condensed_dist_mat = squareform(distance_mat, checks=False)

        if condensed_dist_mat.shape[0] == 0 and len(feature_names_list) > 1:
            logging.warning(
                "  Condensed distance matrix is empty for multiple features. Cannot cluster. Returning all features.")
            return feature_names_list

        linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)

    logging.info(f"  Forming flat clusters with distance threshold: {distance_thresh}...")
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')

    selected_representatives_list = []
    for i in range(1, len(np.unique(cluster_labels_arr)) + 1):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_indices: continue
        if len(cluster_indices) == 1:
            selected_representatives_list.append(feature_names_list[cluster_indices[0]])
        else:
            cluster_member_names = [feature_names_list[idx] for idx in cluster_indices]
            cluster_assoc_submat = assoc_df.loc[cluster_member_names, cluster_member_names]

            if cluster_assoc_submat.empty:
                logging.warning(f"  Empty submatrix for cluster {i}. Skipping representative selection.")
                continue

            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            if sum_abs_assoc_arr.size > 0:
                rep_local_idx = np.argmax(sum_abs_assoc_arr)
                selected_representatives_list.append(cluster_member_names[rep_local_idx])
            else:
                selected_representatives_list.append(cluster_member_names[0])

    return sorted(list(set(selected_representatives_list)))


def run_benchmarking_for_feature_set(x_train_fs, y_train_fs, x_test_fs, y_test_fs, feature_set_label, threshold_value,
                                     run_number):
    logging.info(f"\n--- Benchmarking for: {feature_set_label} (Run {run_number}) ---")

    all_models_results_for_feature_set = []
    # Use fixed RANDOM_STATE for CV splits, as requested
    kf_cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model_template_base in MODELS_TO_BENCHMARK.items():
        logging.info(f"    Tuning model: {model_name}...")
        param_grid_fs = PARAM_GRIDS_FOR_MODELS.get(model_name, {})

        # Set fixed RANDOM_STATE for stochastic models during instantiation, as requested
        model_template_fs = model_template_base
        if hasattr(model_template_fs, 'random_state'):
            # Create a new instance with fixed random_state to ensure consistency across runs
            # This is important if GridSearchCV's refit=True or if the model itself is stochastic
            model_template_fs = model_template_base.__class__(
                **{k: v for k, v in model_template_base.get_params().items() if k != 'random_state'})
            model_template_fs.set_params(random_state=RANDOM_STATE)

        grid_search_fs = GridSearchCV(
            estimator=model_template_fs, param_grid=param_grid_fs, scoring=GRIDSEARCH_SCORING_METRIC,
            cv=kf_cv, n_jobs=-1, verbose=0, error_score='raise'
        )
        current_model_result_row = {"Model": model_name, "Feature Set Name": feature_set_label,
                                    "Number of Features": x_train_fs.shape[1], "Threshold Value": threshold_value,
                                    "Run Number": run_number}
        try:
            grid_search_fs.fit(x_train_fs, y_train_fs)
            best_estimator_fs = grid_search_fs.best_estimator_

            logging.info(f"    Best params found: {grid_search_fs.best_params_}")
            current_model_result_row["Best Params"] = str(grid_search_fs.best_params_)

            cv_results_for_model_fs = cross_validate(
                best_estimator_fs, x_train_fs, y_train_fs, cv=kf_cv,
                scoring=CV_SCORING_REPORT_DICT, n_jobs=-1
            )

            for key_metric, scorer_name in CV_SCORING_REPORT_DICT.items():
                current_model_result_row[f"Mean CV {key_metric.replace('_', ' ').title()}"] = np.mean(
                    cv_results_for_model_fs.get(f'test_{key_metric}', [np.nan]))

            # Calculate test set metrics
            y_pred_test_fs = best_estimator_fs.predict(x_test_fs)
            current_model_result_row["Test Accuracy"] = accuracy_score(y_test_fs, y_pred_test_fs)
            current_model_result_row["Test F1 Weighted"] = f1_score(y_test_fs, y_pred_test_fs, average='weighted',
                                                                    zero_division=0)
            current_model_result_row["Test F1 Macro"] = f1_score(y_test_fs, y_pred_test_fs, average='macro',
                                                                 zero_division=0)
            current_model_result_row["Error"] = None

        except Exception as e_gs:
            logging.error(f"      ERROR: Model {model_name} failed: {e_gs}", exc_info=True)
            current_model_result_row["Error"] = str(e_gs)
            # Fill other metric fields with NaN if an error occurred
            for key_metric, scorer_name in CV_SCORING_REPORT_DICT.items():
                current_model_result_row[f"Mean CV {key_metric.replace('_', ' ').title()}"] = np.nan
            current_model_result_row["Test Accuracy"] = np.nan
            current_model_result_row["Test F1 Weighted"] = np.nan
            current_model_result_row["Test F1 Macro"] = np.nan

        all_models_results_for_feature_set.append(current_model_result_row)

    if not all_models_results_for_feature_set:
        logging.warning(f"    No models successfully tuned for {feature_set_label}.")
        return pd.DataFrame()  # Return empty DataFrame

    return pd.DataFrame(all_models_results_for_feature_set)


# --- Main Orchestration ---
def main():
    logging.info(f"--- Starting Script: 4_clustering.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    # Load pre-split and preprocessed data once at the beginning
    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")
    y_train_orig = load_pickle_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_pickle_data(TEST_X_PATH, "original X_test")
    y_test_orig = load_pickle_data(TEST_Y_PATH, "original y_test")

    if any(data is None for data in [X_train_orig, y_train_orig, X_test_orig, y_test_orig]):
        logging.error("Exiting due to missing data.")
        return

    # Ensure X_train_orig and X_test_orig are DataFrames with sanitized columns if they are not already.
    X_train_sanitized = sanitize_feature_names_df(pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names_df(pd.DataFrame(X_test_orig)).reindex(columns=X_train_sanitized.columns,
                                                                                    fill_value=0)

    # Ensure target variables are flat numpy arrays for model compatibility
    y_train_ravel = y_train_orig.ravel()
    y_test_ravel = y_test_orig.ravel()

    all_runs_combined_df = pd.DataFrame()  # This will store results from all runs

    # Pre-calculate association matrix and linkage matrix for the fixed data
    precomputed_association_matrix = calculate_association_df(X_train_sanitized)
    distance_m = 1 - np.abs(precomputed_association_matrix.values)
    np.fill_diagonal(distance_m, 0)

    precomputed_linkage = None
    if distance_m.shape[0] > 1:
        try:
            # Fix: Ensure non-negative distances before squareform
            distance_m = np.maximum(0, distance_m)
            condensed_dist_mat = squareform(distance_m, checks=False)
            if condensed_dist_mat.shape[0] > 0:
                precomputed_linkage = hierarchy.linkage(condensed_dist_mat, method=CLUSTERING_LINKAGE_METHOD)
            else:
                logging.warning(
                    "  Condensed distance matrix is empty after squareform. Skipping linkage precomputation.")
        except ValueError as e:
            logging.error(
                f"  Error creating condensed distance matrix or linkage: {e}. Skipping clustering precomputation.",
                exc_info=True)

    for run_idx in range(N_REPEATS_ANALYSIS):
        logging.info(f"\n===== Starting Repetition Run {run_idx + 1}/{N_REPEATS_ANALYSIS} =====")

        current_run_all_performance_results_df = pd.DataFrame()  # Accumulate results for this run

        for threshold in CLUSTERING_THRESHOLDS_TO_TEST:
            if threshold is None:
                feature_set_label_base = "Original Features"
                num_features = X_train_sanitized.shape[1]
                current_X_train_fs = X_train_sanitized
                current_X_test_fs = X_test_sanitized
            else:
                feature_set_label_base = f"Clustered (Thresh={threshold})"
                selected_features = get_selected_features_by_clustering(
                    X_train_sanitized, threshold, CLUSTERING_LINKAGE_METHOD,
                    precomputed_assoc_df=precomputed_association_matrix, precomputed_linkage_matrix=precomputed_linkage
                )
                num_features = len(selected_features)

                if not selected_features:
                    logging.warning(f"  No features selected for threshold {threshold} in run {run_idx + 1}. Skipping.")
                    continue

                current_X_train_fs = X_train_sanitized[selected_features]
                current_X_test_fs = X_test_sanitized[selected_features]

            feature_set_label = f"{feature_set_label_base} ({num_features} feats)"

            # Call the benchmarking function, passing the run_idx
            results_for_current_feature_set = run_benchmarking_for_feature_set(
                current_X_train_fs, y_train_ravel, current_X_test_fs, y_test_ravel, feature_set_label, threshold,
                run_idx + 1
            )

            if not results_for_current_feature_set.empty:
                current_run_all_performance_results_df = pd.concat(
                    [current_run_all_performance_results_df, results_for_current_feature_set], ignore_index=True)

        # Append results of the current run to the overall list
        if not current_run_all_performance_results_df.empty:
            all_runs_combined_df = pd.concat([all_runs_combined_df, current_run_all_performance_results_df],
                                             ignore_index=True)

    logging.info("\n\n===== Overall Performance Summary (All Runs, Models, and Feature Sets) =====")
    # The all_runs_combined_df now contains results from all repetitions.
    # The plot will automatically compute means and standard deviations from this data.

    # You might still want a summary table of means/stds for logging
    summary_for_logging_df = all_runs_combined_df.groupby(['Model', 'Feature Set Name', 'Threshold Value'])[
        METRIC_FOR_FINAL_COMPARISON_PLOT].agg(['mean', 'std']).reset_index()
    summary_for_logging_df.rename(columns={'mean': METRIC_FOR_FINAL_COMPARISON_PLOT + ' (Mean)',
                                           'std': METRIC_FOR_FINAL_COMPARISON_PLOT + ' (Std)'}, inplace=True)

    logging.info(f"\n{summary_for_logging_df.to_string(index=False)}")

    # Find and save the best threshold (based on overall mean performance of the best model across runs)
    # This logic now uses the summary_for_logging_df which has means.
    best_model_per_feature_set_mean = summary_for_logging_df.loc[
        summary_for_logging_df.groupby('Feature Set Name')[METRIC_FOR_FINAL_COMPARISON_PLOT + ' (Mean)'].idxmax()]
    best_threshold_row_overall = \
    best_model_per_feature_set_mean.sort_values(by=METRIC_FOR_FINAL_COMPARISON_PLOT + ' (Mean)', ascending=False).iloc[
        0]
    best_threshold_overall = best_threshold_row_overall['Threshold Value']
    logging.info(
        f"\nOverall best performing threshold (based on mean {METRIC_FOR_FINAL_COMPARISON_PLOT}): {best_threshold_overall}")
    with open(BEST_THRESHOLD_FILE, 'w') as f:
        json.dump({'best_threshold': best_threshold_overall}, f)
    logging.info(f"Best threshold saved to: {BEST_THRESHOLD_FILE}")

    # Identify and save the top N models (overall best performing models across all feature sets, based on mean)
    top_models_overall_mean = \
    summary_for_logging_df.sort_values(by=METRIC_FOR_FINAL_COMPARISON_PLOT + ' (Mean)', ascending=False)[
        'Model'].unique()[:N_TOP_MODELS_TO_SAVE]
    logging.info(
        f"Top {N_TOP_MODELS_TO_SAVE} models to be passed to next script (overall mean): {top_models_overall_mean}")
    with open(TOP_MODELS_FILE, 'w') as f:
        json.dump({'top_models': list(top_models_overall_mean)}, f)
    logging.info(f"Top models saved to: {TOP_MODELS_FILE}")

    # --- Plotting All Models vs. Clustering Cases with Error Bars ---
    plt.figure(figsize=(20, 12))  # Adjust figure size for better readability

    # Define order for x-axis to ensure consistency (use unique Feature Set Names from the combined data)
    x_order = all_runs_combined_df['Feature Set Name'].unique().tolist()

    sns.barplot(
        x="Feature Set Name",
        y=METRIC_FOR_FINAL_COMPARISON_PLOT,  # Use the raw metric column name for automatic mean/std plotting
        hue="Model",
        data=all_runs_combined_df,  # Use the full combined data frame for automatic error bars
        palette="viridis",
        errorbar="sd",  # Show standard deviation as error bars
        order=x_order
    )
    plt.title(
        f'Model {METRIC_FOR_FINAL_COMPARISON_PLOT} Comparison Across All Clustering Cases (Mean & Std Dev over {N_REPEATS_ANALYSIS} runs)',
        fontsize=16)
    plt.xlabel("Feature Set Configuration", fontsize=12)
    plt.ylabel(f"{METRIC_FOR_FINAL_COMPARISON_PLOT} (Mean)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)  # Rotate labels for better fit
    plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')  # Move legend outside
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend

    chart_save_path = os.path.join(BASE_RESULTS_DIR,
                                   f"all_models_clustering_{METRIC_FOR_FINAL_COMPARISON_PLOT.replace(' ', '_').lower()}_repeated_comparison.png")
    plt.savefig(chart_save_path)
    logging.info(f"\nComparison bar chart with error bars saved to: {chart_save_path}")
    plt.show()

    detailed_results_path = os.path.join(BASE_RESULTS_DIR,
                                         "clustering_performance_detailed_results_all_models_repeated.csv")
    all_runs_combined_df.to_csv(detailed_results_path, index=False, float_format='%.6f')
    logging.info(f"Detailed performance results for all models and runs saved to: {detailed_results_path}")


if __name__ == '__main__':
    main()