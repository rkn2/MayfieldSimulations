import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re
import ast

from sklearn.metrics import make_scorer, f1_score
from sklearn.inspection import permutation_importance
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from sklearn.linear_model import LogisticRegression

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
BASE_RESULTS_DIR = 'cluster_importance_results_final'
CLUSTER_LABELS_DIR = 'cluster_exploration_results'

USER_DEFINED_CLUSTERING_THRESHOLD = 0.5
CLUSTERING_LINKAGE_METHOD = 'average'
SCORING_METRIC_FOR_IMPORTANCE = 'f1_weighted'
N_TOP_CLUSTERS_TO_PLOT = 20
RANDOM_STATE = 42
N_PERMUTATION_REPEATS = 10  # Number of repeats for permutation importance

# --- Paths ---
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
CV_RESULTS_CSV_PATH = os.path.join(DATA_DIR, 'model_tuned_cv_results.csv')
CLUSTER_LABELS_CSV_PATH = os.path.join(CLUSTER_LABELS_DIR, f"threshold_{USER_DEFINED_CLUSTERING_THRESHOLD}",
                                       "cluster_labels_and_contributing_features_thresh0.5.csv")

# Define the top models to evaluate
TOP_MODELS_TO_TEST = {
    "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1) if LGBM_AVAILABLE else None,
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss') if XGB_AVAILABLE else None,
}
# Filter out any models that failed to import
TOP_MODELS_TO_TEST = {k: v for k, v in TOP_MODELS_TO_TEST.items() if v is not None}


# --- Helper Functions ---
def load_data(file_path, description="data"):
    print(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        print(f"  Successfully loaded: {description}")
        return data
    except FileNotFoundError:
        print(f"Error: {description} file not found at {file_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}. Exiting.")
        exit()


def sanitize_feature_names_df(df):
    if not isinstance(df, pd.DataFrame): return df
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]
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


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    feature_names_list = original_df.columns.tolist()
    if len(feature_names_list) <= 1: return feature_names_list
    print("  Calculating association matrix for clustering...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_df = associations(original_df, nom_nom_assoc='cramer', compute_only=True)['corr'].fillna(0)
    print("  Performing hierarchical clustering...")
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)
    if condensed_dist_mat.shape[0] == 0: return feature_names_list
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    selected_representatives_list = []
    for i in range(1, len(np.unique(cluster_labels_arr)) + 1):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if len(cluster_indices) == 1:
            selected_representatives_list.append(feature_names_list[cluster_indices[0]])
        else:
            sum_abs_assoc = np.abs(assoc_df.iloc[cluster_indices, cluster_indices].values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc)
            selected_representatives_list.append(feature_names_list[cluster_indices[rep_local_idx]])
    return sorted(list(set(selected_representatives_list)))


def load_best_parameters(params_csv_path):
    print(f"\nLoading best parameters from: {params_csv_path}")
    try:
        params_df = pd.read_csv(params_csv_path)
        model_best_params = {}
        for _, row in params_df.iterrows():
            model_name = row['Model']
            try:
                params_str = row['Best Params']
                if pd.isna(params_str) or params_str.lower() in ['n/a', '{}']:
                    model_best_params[model_name] = {}
                else:
                    best_params_dict = ast.literal_eval(params_str)
                    model_best_params[model_name] = best_params_dict
            except Exception as e:
                print(f"  Warning: Could not parse params for {model_name}: {e}. Using defaults.")
                model_best_params[model_name] = {}
        print("  Best parameters loaded successfully.")
        return model_best_params
    except FileNotFoundError:
        print(f"ERROR: Best parameters CSV file not found at {params_csv_path}.")
        return None
    except Exception as e:
        print(f"ERROR: Could not load best parameters from CSV: {e}")
        return None


# --- Main Orchestration ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    results_subdir = os.path.join(BASE_RESULTS_DIR, f"thresh_{USER_DEFINED_CLUSTERING_THRESHOLD}")
    os.makedirs(results_subdir, exist_ok=True)

    print(f"Starting Final Cluster Importance Analysis for THRESHOLD = {USER_DEFINED_CLUSTERING_THRESHOLD}")

    X_train_orig = load_data(TRAIN_X_PATH, "original X_train")
    y_train = load_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_data(TEST_X_PATH, "original X_test")
    y_test = load_data(TEST_Y_PATH, "original y_test")

    all_best_params = load_best_parameters(CV_RESULTS_CSV_PATH)
    if all_best_params is None:
        print("Exiting due to failure in loading model parameters.")
        exit()

    X_train_sanitized = sanitize_feature_names_df(
        X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names_df(
        X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else pd.DataFrame(X_test_orig))
    X_test_sanitized = X_test_sanitized.reindex(columns=X_train_sanitized.columns, fill_value=0)

    y_train_ravel = y_train.values.ravel() if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.ravel()
    y_test_ravel = y_test.values.ravel() if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test.ravel()

    print(f"\nPerforming feature selection with optimal clustering threshold: {USER_DEFINED_CLUSTERING_THRESHOLD}...")
    selected_features = get_selected_features_by_clustering(X_train_sanitized, USER_DEFINED_CLUSTERING_THRESHOLD,
                                                            CLUSTERING_LINKAGE_METHOD)
    if not selected_features:
        print("Error: No features selected by clustering. Cannot proceed.")
        return
    print(f"  Number of representative features (clusters) selected: {len(selected_features)}")

    X_train_selected = X_train_sanitized[selected_features]
    X_test_selected = X_test_sanitized[selected_features]

    print(f"Loading cluster labels from {CLUSTER_LABELS_CSV_PATH}...")
    try:
        labels_df = pd.read_csv(CLUSTER_LABELS_CSV_PATH)
        cluster_label_map = pd.Series(labels_df['Cluster Label'].values,
                                      index=labels_df['Representative Feature']).to_dict()
        print("  Successfully created cluster label map.")
    except Exception as e:
        print(f"ERROR: Could not load or process cluster labels CSV: {e}. Raw feature names will be used.")
        cluster_label_map = {}

    all_model_importances = []

    for model_name, model_template in TOP_MODELS_TO_TEST.items():
        print(f"\n--- Analyzing Importance for Model: {model_name} ---")

        model_params = all_best_params.get(model_name, {})
        model_instance = model_template.set_params(**model_params)

        print(f"  Retraining {model_name} on {len(selected_features)} clustered features...")
        model_instance.fit(X_train_selected, y_train_ravel)

        print(f"  Calculating Permutation Importance for {model_name}...")
        scorer = make_scorer(f1_score, average='weighted', zero_division=0)
        perm_importance_result = permutation_importance(
            model_instance, X_test_selected, y_test_ravel, scoring=scorer,
            n_repeats=N_PERMUTATION_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
        )

        for i, rep_feature_name in enumerate(selected_features):
            descriptive_label = cluster_label_map.get(rep_feature_name, rep_feature_name)

            ### NEW: Calculate p-value ###
            # Count how many times the permuted score drop was <= 0
            count_le_zero = np.sum(perm_importance_result.importances[i] <= 0)
            # Estimate p-value
            p_value = (count_le_zero + 1) / (N_PERMUTATION_REPEATS + 1)

            all_model_importances.append({
                'Cluster Label': descriptive_label,
                'Model': model_name,
                'Importance (Mean Drop)': perm_importance_result.importances_mean[i],
                'Importance (Std Dev)': perm_importance_result.importances_std[i],
                'p-value': p_value
            })

    importances_df = pd.DataFrame(all_model_importances)

    ### NEW: Add significance column ###
    importances_df['Significant (p<0.05)'] = importances_df['p-value'] < 0.05

    max_importance_per_cluster = importances_df.groupby('Cluster Label')['Importance (Mean Drop)'].max()
    top_n_labels = max_importance_per_cluster.sort_values(ascending=False).head(N_TOP_CLUSTERS_TO_PLOT).index

    top_n_df = importances_df[importances_df['Cluster Label'].isin(top_n_labels)]

    plt.figure(figsize=(14, 12))
    sns.barplot(x='Importance (Mean Drop)', y='Cluster Label', hue='Model', data=top_n_df, palette='viridis',
                order=top_n_labels)
    plt.title(
        f'Top {N_TOP_CLUSTERS_TO_PLOT} Most Important Feature Clusters (Threshold={USER_DEFINED_CLUSTERING_THRESHOLD})',
        fontsize=16, pad=20)
    plt.xlabel(f"Mean Drop in Test {SCORING_METRIC_FOR_IMPORTANCE.replace('_', ' ').title()}", fontsize=12)
    plt.ylabel("Feature Cluster", fontsize=12)
    plt.legend(title='Model')
    plt.tight_layout()

    plot_save_path = os.path.join(results_subdir, "final_top_cluster_importances_multi_model.png")
    plt.savefig(plot_save_path)
    print(f"\nFinal multi-model importance plot saved to: {plot_save_path}")
    plt.show()

    print("\n--- Full Cluster Importance Results ---")
    # Use to_string() to ensure the full DataFrame is printed without truncation
    # ### MODIFIED: Ensure new columns are included in the printout ###
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(importances_df.sort_values(by=['Model', 'Importance (Mean Drop)'], ascending=[True, False]).to_string())


if __name__ == '__main__':
    main()

