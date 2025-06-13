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
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
RANDOM_STATE = 42
N_SPLITS_CV = 5
CLUSTERING_THRESHOLDS_TO_TEST = [None, 0.8, 0.9]
CLUSTERING_LINKAGE_METHOD = 'average'
GRIDSEARCH_SCORING_METRIC = 'f1_weighted'
PRIMARY_CV_METRIC_NAME_FOR_BEST_MODEL = f"Mean CV {GRIDSEARCH_SCORING_METRIC.replace('_', ' ').title()}"
METRIC_FOR_FINAL_COMPARISON_PLOT = 'Test F1 Weighted'
N_TOP_MODELS_TO_SAVE = 3  # Number of top models to pass to the next script (This variable will now be ignored for top_models selection based on threshold)

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
        assoc_results = associations(dataframe, nom_nom_assoc='cramer', compute_only=True, mark_columns=False)
    association_dataframe = assoc_results['corr']
    logging.info("  Association matrix calculated.")
    return association_dataframe.fillna(0)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth, precomputed_assoc_df=None,
                                        precomputed_linkage_matrix=None):
    feature_names_list = original_df.columns.tolist()
    assoc_df = precomputed_assoc_df if precomputed_assoc_df is not None else calculate_association_df(
        original_df.copy())

    if precomputed_linkage_matrix is not None:
        linked = precomputed_linkage_matrix
    else:
        logging.info(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
        distance_mat = 1 - np.abs(assoc_df.values)
        np.fill_diagonal(distance_mat, 0)
        condensed_dist_mat = squareform(distance_mat, checks=False)
        if condensed_dist_mat.shape[0] == 0:
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
            cluster_assoc_submat = assoc_df.iloc[cluster_indices, cluster_indices]
            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc_arr)
            selected_representatives_list.append(feature_names_list[cluster_indices[rep_local_idx]])

    return sorted(list(set(selected_representatives_list)))


def run_benchmarking_for_feature_set(x_train_fs, y_train_fs, x_test_fs, y_test_fs, feature_set_name_with_count,
                                     threshold_value):
    logging.info(f"\n--- Benchmarking for: {feature_set_name_with_count} ---")

    all_models_results_for_feature_set = []
    kf_cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)

    for model_name, model_template_fs in MODELS_TO_BENCHMARK.items():
        logging.info(f"    Tuning model: {model_name}...")
        param_grid_fs = PARAM_GRIDS_FOR_MODELS.get(model_name, {})

        grid_search_fs = GridSearchCV(
            estimator=model_template_fs, param_grid=param_grid_fs, scoring=GRIDSEARCH_SCORING_METRIC,
            cv=kf_cv, n_jobs=-1, verbose=0, error_score='raise'
        )
        current_model_result_row = {"Model": model_name, "Feature Set Name": feature_set_name_with_count,
                                    "Number of Features": x_train_fs.shape[1], "Threshold Value": threshold_value}
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
        logging.warning(f"    No models successfully tuned for {feature_set_name_with_count}.")
        return pd.DataFrame()  # Return empty DataFrame

    return pd.DataFrame(all_models_results_for_feature_set)


# --- Main Orchestration ---
def main():
    logging.info(f"--- Starting Script: 4_clustering.py ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")
    y_train_orig = load_pickle_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_pickle_data(TEST_X_PATH, "original X_test")
    y_test_orig = load_pickle_data(TEST_Y_PATH, "original y_test")

    if any(data is None for data in [X_train_orig, y_train_orig, X_test_orig, y_test_orig]):
        logging.error("Exiting due to missing data.")
        return

    X_train_sanitized = sanitize_feature_names_df(pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names_df(pd.DataFrame(X_test_orig)).reindex(columns=X_train_sanitized.columns,
                                                                                    fill_value=0)

    y_train_ravel = y_train_orig.ravel()
    y_test_ravel = y_test_orig.ravel()

    all_performance_results_df = pd.DataFrame()  # This will store all results

    precomputed_association_matrix = calculate_association_df(X_train_sanitized)
    distance_m = 1 - np.abs(precomputed_association_matrix.values)
    np.fill_diagonal(distance_m, 0)
    precomputed_linkage = hierarchy.linkage(squareform(distance_m, checks=False), method=CLUSTERING_LINKAGE_METHOD)

    for threshold in CLUSTERING_THRESHOLDS_TO_TEST:
        if threshold is None:
            feature_set_label_base = "Original Features"
            num_features = X_train_sanitized.shape[1]
            current_X_train = X_train_sanitized
            current_X_test = X_test_sanitized
        else:
            feature_set_label_base = f"Clustered (Thresh={threshold})"
            selected_features = get_selected_features_by_clustering(
                X_train_sanitized, threshold, CLUSTERING_LINKAGE_METHOD,
                precomputed_assoc_df=precomputed_association_matrix, precomputed_linkage_matrix=precomputed_linkage
            )
            num_features = len(selected_features)
            current_X_train = X_train_sanitized[selected_features]
            current_X_test = X_test_sanitized[selected_features]

        feature_set_label = f"{feature_set_label_base} ({num_features} feats)"

        # Call the modified benchmarking function
        results_for_current_feature_set = run_benchmarking_for_feature_set(
            current_X_train, y_train_ravel, current_X_test, y_test_ravel, feature_set_label, threshold
        )

        if not results_for_current_feature_set.empty:
            all_performance_results_df = pd.concat([all_performance_results_df, results_for_current_feature_set],
                                                   ignore_index=True)

    logging.info("\n\n===== Overall Performance Summary (All Models and Feature Sets) =====")
    logging.info(f"\n{all_performance_results_df.to_string(index=False)}")

    # Find and save the best threshold (still based on the best model for each threshold, but we have all data now)
    # This logic assumes we want the overall best threshold determined by the best model's performance
    # which is likely the intent for proceeding scripts.
    best_threshold_selection_df = all_performance_results_df.loc[
        all_performance_results_df.groupby('Feature Set Name')[METRIC_FOR_FINAL_COMPARISON_PLOT].idxmax()]
    best_threshold_row = \
    best_threshold_selection_df.sort_values(by=METRIC_FOR_FINAL_COMPARISON_PLOT, ascending=False).iloc[0]
    best_threshold = best_threshold_row['Threshold Value']
    logging.info(
        f"\nBest performing threshold based on overall best model's '{METRIC_FOR_FINAL_COMPARISON_PLOT}': {best_threshold}")
    with open(BEST_THRESHOLD_FILE, 'w') as f:
        json.dump({'best_threshold': best_threshold}, f)
    logging.info(f"Best threshold saved to: {BEST_THRESHOLD_FILE}")

    # Identify and save the top N models (overall best performing models across all feature sets)
    top_models_overall = all_performance_results_df.sort_values(by=METRIC_FOR_FINAL_COMPARISON_PLOT, ascending=False)[
                             'Model'].unique()[:N_TOP_MODELS_TO_SAVE]
    logging.info(f"Top {N_TOP_MODELS_TO_SAVE} models to be passed to next script (overall): {top_models_overall}")
    with open(TOP_MODELS_FILE, 'w') as f:
        json.dump({'top_models': list(top_models_overall)}, f)
    logging.info(f"Top models saved to: {TOP_MODELS_FILE}")

    # --- Plotting All Models vs. Clustering Cases ---
    plt.figure(figsize=(18, 10))  # Adjust figure size for better readability
    sns.barplot(x="Feature Set Name", y=METRIC_FOR_FINAL_COMPARISON_PLOT, hue="Model", data=all_performance_results_df,
                palette="viridis")
    plt.title(f'Model {METRIC_FOR_FINAL_COMPARISON_PLOT} Comparison Across All Clustering Cases', fontsize=16)
    plt.xlabel("Feature Set Configuration", fontsize=12)
    plt.ylabel(METRIC_FOR_FINAL_COMPARISON_PLOT, fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)  # Rotate labels for better fit
    plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left')  # Move legend outside
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for the legend

    chart_save_path = os.path.join(BASE_RESULTS_DIR,
                                   f"all_models_clustering_{METRIC_FOR_FINAL_COMPARISON_PLOT.replace(' ', '_').lower()}_comparison.png")
    plt.savefig(chart_save_path)
    logging.info(f"\nComparison bar chart saved to: {chart_save_path}")
    plt.show()

    detailed_results_path = os.path.join(BASE_RESULTS_DIR, "clustering_performance_detailed_results_all_models.csv")
    all_performance_results_df.to_csv(detailed_results_path, index=False, float_format='%.6f')
    logging.info(f"Detailed performance results for all models saved to: {detailed_results_path}")


if __name__ == '__main__':
    main()