import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns  # Still useful for palettes
import re
import ast  # For safely evaluating string representations of dictionaries

# --- Scikit-learn ---
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer
)
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier

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
BASE_RESULTS_DIR = 'cluster_importance_results_loaded_params'  # New results directory
BEST_PARAMS_CSV_PATH = os.path.join(DATA_DIR, 'model_tuned_cv_results.csv')  # Path to CSV from your benchmarking script

TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

# --- USER-DEFINED CLUSTERING THRESHOLD ---
USER_DEFINED_CLUSTERING_THRESHOLD = 0.7
N_TOP_CLUSTERS_TO_PLOT = 10
RANDOM_STATE = 42
# N_SPLITS_CV = 5 # Not needed for GridSearchCV in this script anymore

CLUSTERING_LINKAGE_METHOD = 'average'
SCORING_METRIC_FOR_IMPORTANCE = 'f1_weighted'  # Metric for permutation importance

# Define all models you want to evaluate.
# Names should match the 'Model' column in your BEST_PARAMS_CSV_PATH
MODELS_TO_EVALUATE = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    # Solver will be set from params
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),  # Renamed for consistency
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),  # Added back
}
if XGB_AVAILABLE:
    MODELS_TO_EVALUATE["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE,
                                                      use_label_encoder=False)  # eval_metric from params
if LGBM_AVAILABLE:
    MODELS_TO_EVALUATE["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)


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


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    feature_names_list = original_df.columns.tolist()
    if len(feature_names_list) <= 1:
        return feature_names_list

    assoc_df = calculate_association_df(original_df.copy())

    print(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    distance_mat = (distance_mat + distance_mat.T) / 2
    condensed_dist_mat = squareform(distance_mat, checks=False)

    if condensed_dist_mat.shape[0] == 0:
        print("  Warning: Condensed distance matrix is empty with multiple features. Returning all features.")
        return feature_names_list

    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    print(f"  Forming flat clusters with distance threshold: {distance_thresh}...")
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    num_unique_clusters = len(np.unique(cluster_labels_arr))

    selected_representatives_list = []
    for i in range(1, num_unique_clusters + 1):
        cluster_member_indices_list = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_member_indices_list: continue
        if len(cluster_member_indices_list) == 1:
            selected_representatives_list.append(feature_names_list[cluster_member_indices_list[0]])
        else:
            cluster_assoc_submat = assoc_df.iloc[cluster_member_indices_list, cluster_member_indices_list]
            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc_arr)
            rep_original_idx = cluster_member_indices_list[rep_local_idx]
            selected_representatives_list.append(feature_names_list[rep_original_idx])
    return sorted(list(set(selected_representatives_list)))


def load_best_parameters(params_csv_path):
    """Loads best parameters from the CSV file generated by the benchmarking script."""
    print(f"\nLoading best parameters from: {params_csv_path}")
    try:
        params_df = pd.read_csv(params_csv_path)
        model_best_params = {}
        for _, row in params_df.iterrows():
            model_name = row['Model']
            try:
                # The 'Best Params' column is a string representation of a dict
                params_str = row['Best Params']
                if pd.isna(params_str) or params_str.lower() == 'n/a' or params_str == '{}':
                    model_best_params[model_name] = {}  # Default if no params or error
                    print(f"  No valid parameters found for {model_name}, will use defaults.")
                else:
                    # Safely evaluate the string to a dictionary
                    best_params_dict = ast.literal_eval(params_str)
                    model_best_params[model_name] = best_params_dict
            except Exception as e:
                print(f"  Warning: Could not parse params for {model_name}: {e}. Using default params.")
                model_best_params[model_name] = {}  # Fallback to default
        print("Best parameters loaded.")
        return model_best_params
    except FileNotFoundError:
        print(f"Error: Best parameters CSV file not found at {params_csv_path}.")
        return None
    except Exception as e:
        print(f"Error loading best parameters: {e}")
        return None


# --- Main Orchestration ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    current_clustering_threshold = USER_DEFINED_CLUSTERING_THRESHOLD
    results_subdir = os.path.join(BASE_RESULTS_DIR, f"thresh_{current_clustering_threshold}")
    os.makedirs(results_subdir, exist_ok=True)

    print(
        f"Starting Multi-Model Cluster Importance (with Loaded Params) for THRESHOLD = {current_clustering_threshold}")

    # --- 0. Load Pre-Tuned Best Parameters ---
    all_best_params = load_best_parameters(BEST_PARAMS_CSV_PATH)
    if all_best_params is None:
        print("Could not load best parameters. Exiting.")
        return

    # --- 1. Load Data ---
    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")
    y_train_orig = load_pickle_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_pickle_data(TEST_X_PATH, "original X_test")
    y_test_orig = load_pickle_data(TEST_Y_PATH, "original y_test")

    if X_train_orig is None or y_train_orig is None or X_test_orig is None or y_test_orig is None:
        print("Exiting due to missing data.")
        return

    X_train_sanitized = sanitize_feature_names_df(
        X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names_df(
        X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else pd.DataFrame(X_test_orig))

    if isinstance(X_train_sanitized, pd.DataFrame) and isinstance(X_test_sanitized, pd.DataFrame):
        train_cols = X_train_sanitized.columns
        X_test_sanitized = X_test_sanitized.reindex(columns=train_cols, fill_value=0)

    y_train_ravel = y_train_orig.values.ravel() if isinstance(y_train_orig,
                                                              (pd.Series, pd.DataFrame)) else y_train_orig.ravel()
    y_test_ravel = y_test_orig.values.ravel() if isinstance(y_test_orig,
                                                            (pd.Series, pd.DataFrame)) else y_test_orig.ravel()

    # --- 2. Feature Selection ---
    if not isinstance(X_train_sanitized, pd.DataFrame) or X_train_sanitized.shape[1] <= 1:
        print(
            "Warning: X_train is not a suitable DataFrame for clustering or has too few features. Using all features.")
        selected_features = X_train_sanitized.columns.tolist() if isinstance(X_train_sanitized, pd.DataFrame) else [
            f"feature_{i}" for i in range(X_train_sanitized.shape[1])]
    else:
        print(f"\nPerforming feature selection with clustering threshold: {current_clustering_threshold}...")
        selected_features = get_selected_features_by_clustering(
            X_train_sanitized,
            current_clustering_threshold,
            CLUSTERING_LINKAGE_METHOD
        )

    if not selected_features:
        print("Error: No features selected by clustering. Cannot proceed.")
        return

    num_selected_features = len(selected_features)
    print(f"Number of features (clusters) selected: {num_selected_features}")
    if num_selected_features == 0:
        print("No features selected, stopping analysis.")
        return

    if isinstance(X_train_sanitized, pd.DataFrame):
        X_train_selected = X_train_sanitized[selected_features]
        X_test_selected = X_test_sanitized[selected_features]
    else:
        print("Warning: Feature selection on NumPy array requires careful index mapping.")
        X_train_selected = X_train_sanitized
        X_test_selected = X_test_sanitized

    # --- 3. Train Models with Loaded Parameters (No GridSearchCV) ---
    print("\n--- Training models with pre-loaded best parameters ---")
    trained_models = {}

    for model_name, model_template in MODELS_TO_EVALUATE.items():
        print(f"  Training model: {model_name}...")
        model_params = all_best_params.get(model_name, {})  # Get params for this model

        # Create a new instance to ensure fresh state
        current_model = MODELS_TO_EVALUATE[model_name]  # This gets the template

        # Some models might have parameters that are not settable after init (e.g. n_jobs for RF)
        # or specific ways to handle them (e.g. eval_metric for XGB).
        # For simplicity, we try set_params. More robust handling might be needed for complex cases.
        try:
            # Filter out params not applicable to the current model instance to avoid errors
            valid_params = {k: v for k, v in model_params.items() if hasattr(current_model, k)}

            # Special handling for XGBoost eval_metric if it was in params
            # and not a direct parameter of XGBClassifier constructor used in MODELS_TO_EVALUATE
            if model_name == "XGBoost" and 'eval_metric' in model_params and 'eval_metric' not in valid_params:
                # eval_metric is often passed to .fit() for XGB, not set_params directly on classifier object
                # if it's not a constructor param. We'll assume it was handled by the original tuning.
                pass  # For now, we rely on the constructor or default.

            if valid_params:
                current_model.set_params(**valid_params)
                print(f"    Set parameters for {model_name}: {valid_params}")
            else:
                print(
                    f"    Using default parameters for {model_name} as no valid pre-tuned params were found/applicable.")

            current_model.fit(X_train_selected, y_train_ravel)
            trained_models[model_name] = current_model
            print(f"    Successfully trained {model_name}.")
        except Exception as e:
            print(f"    ERROR: Model {model_name} failed during training with loaded params: {e}")
            trained_models[model_name] = None
            continue

    if not any(trained_models.values()):
        print("No models successfully trained. Cannot proceed with feature importance.")
        return

    # --- 4. Calculate Permutation Importance for EACH trained model ---
    all_model_importances_list = []
    print(f"\nCalculating Permutation Importance for selected features using all trained models...")

    if SCORING_METRIC_FOR_IMPORTANCE == 'f1_weighted':
        perm_scorer = make_scorer(f1_score, average='weighted', zero_division=0)
    elif SCORING_METRIC_FOR_IMPORTANCE == 'accuracy':
        perm_scorer = 'accuracy'
    else:
        print(
            f"Warning: Permutation importance using default or potentially misconfigured scorer for '{SCORING_METRIC_FOR_IMPORTANCE}'.")
        perm_scorer = SCORING_METRIC_FOR_IMPORTANCE

    for model_name, model_instance in trained_models.items():
        if model_instance is None:
            print(f"  Skipping permutation importance for {model_name} as it was not successfully trained.")
            continue
        print(f"  Calculating for model: {model_name}...")
        try:
            perm_importance_result = permutation_importance(
                model_instance,  # Use the model trained with pre-loaded params
                X_test_selected,
                y_test_ravel,
                scoring=perm_scorer,
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            for i, feature_name in enumerate(selected_features):
                all_model_importances_list.append({
                    'Feature (Cluster Representative)': feature_name,
                    'Model': model_name,
                    'Importance (Mean Drop in Score)': perm_importance_result.importances_mean[i],
                    'Importance (Std Dev)': perm_importance_result.importances_std[i]
                })
        except Exception as e_perm:
            print(f"    Error calculating permutation importance for {model_name}: {e_perm}")

    if not all_model_importances_list:
        print("No permutation importance results to visualize.")
        return

    all_importances_df = pd.DataFrame(all_model_importances_list)

    max_importance_per_feature = all_importances_df.groupby('Feature (Cluster Representative)')[
        'Importance (Mean Drop in Score)'].max()
    top_n_feature_names = max_importance_per_feature.sort_values(ascending=False).head(
        N_TOP_CLUSTERS_TO_PLOT).index.tolist()

    top_n_importances_df = all_importances_df[
        all_importances_df['Feature (Cluster Representative)'].isin(top_n_feature_names)]

    top_n_importances_df['Feature (Cluster Representative)'] = pd.Categorical(
        top_n_importances_df['Feature (Cluster Representative)'],
        categories=top_n_feature_names,
        ordered=True
    )
    top_n_importances_df = top_n_importances_df.sort_values('Feature (Cluster Representative)')

    print("\n--- Top Clustered Feature Importances (Across All Models) ---")

    importance_csv_path = os.path.join(results_subdir,
                                       f"multi_model_cluster_importances_thresh_{current_clustering_threshold}.csv")
    all_importances_df.to_csv(importance_csv_path, index=False)
    print(f"\nFull multi-model feature importances saved to: {importance_csv_path}")

    # --- 5. Plotting ---
    plot_df = top_n_importances_df.pivot(
        index='Feature (Cluster Representative)',
        columns='Model',
        values='Importance (Mean Drop in Score)'
    )
    plot_df_std = top_n_importances_df.pivot(
        index='Feature (Cluster Representative)',
        columns='Model',
        values='Importance (Std Dev)'
    )

    plot_df = plot_df.reindex(top_n_feature_names)  # Ensure correct order
    plot_df_std = plot_df_std.reindex(top_n_feature_names)  # Ensure correct order

    n_models = len(plot_df.columns)
    n_features_to_plot = len(plot_df.index)

    bar_height = 0.8 / n_models
    index = np.arange(n_features_to_plot)

    fig, ax = plt.subplots(figsize=(14, max(8, n_features_to_plot * 0.7 * n_models * 0.3)))

    colors = sns.color_palette('viridis', n_colors=n_models)

    for i, model_name_plot in enumerate(plot_df.columns):  # Use a different variable name
        means = plot_df[model_name_plot].values
        stds = plot_df_std[model_name_plot].values

        bar_positions = index - (0.8 - bar_height) / 2 + i * bar_height

        ax.barh(
            bar_positions,
            means,
            bar_height * 0.9,
            xerr=stds,
            label=model_name_plot,
            color=colors[i],
            capsize=3,
            alpha=0.8
        )

    ax.set_xlabel(
        f"Mean Drop in Test {SCORING_METRIC_FOR_IMPORTANCE.replace('_', ' ').title()} (Permutation Importance)",
        fontsize=12)
    ax.set_ylabel("Cluster Representative Feature", fontsize=12)
    ax.set_title(
        f'Top {min(N_TOP_CLUSTERS_TO_PLOT, n_features_to_plot)} Most Important Clusters (Thresh={current_clustering_threshold})',
        fontsize=14)
    ax.set_yticks(index)
    ax.set_yticklabels(plot_df.index)
    ax.invert_yaxis()
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plot_save_path = os.path.join(results_subdir,
                                  f"top_multi_model_cluster_importances_thresh_{current_clustering_threshold}.png")
    plt.savefig(plot_save_path)
    print(f"Multi-model importance plot saved to: {plot_save_path}")
    plt.show()

    print("\nMulti-model cluster importance analysis finished.")


if __name__ == '__main__':
    main()
