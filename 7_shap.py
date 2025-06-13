import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import shap
import logging
import sys
import re
import ast
import seaborn as sns

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Scikit-learn ---
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
try:
    import mord

    MORD_AVAILABLE = True
except ImportError:
    MORD_AVAILABLE = False


# --- Logging Configuration Setup ---
def setup_logging(log_file='pipeline.log'):
    """Sets up logging to both a file and the console."""
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

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
RESULTS_DIR_4 = 'clustering_performance_results'
BASE_RESULTS_DIR = 'shap_results_top_performers'

# --- Paths ---
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
DETAILED_PERFORMANCE_CSV = os.path.join(RESULTS_DIR_4, 'clustering_performance_detailed_results.csv')

# --- Analysis Settings ---
PERFORMANCE_THRESHOLD = 0.8
N_SHAP_SAMPLES = 1000
N_BACKGROUND_SAMPLES = 200
N_TOP_FEATURES_TO_PLOT = 15
CLUSTERING_LINKAGE_METHOD = 'average'


# --- Helper Functions ---
def sanitize_feature_names(df):
    """Sanitizes DataFrame column names to match model's expectations."""
    if not isinstance(df, pd.DataFrame):
        return df
    sanitized_cols = {col: re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns}
    df.rename(columns=sanitized_cols, inplace=True)
    return df


def load_data(file_path, description="data"):
    """Loads pickled data."""
    logging.info(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        logging.info(f"  Successfully loaded: {description}")
        return data
    except FileNotFoundError:
        logging.error(f"Error: {description} file not found at {file_path}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading {description} from {file_path}: {e}", exc_info=True)
        sys.exit(1)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    """Recreates the clustered feature set for a given threshold."""
    if distance_thresh is None or pd.isna(distance_thresh):
        logging.info("  Threshold is None. Using all original features.")
        return original_df.columns.tolist()

    feature_names_list = original_df.columns.tolist()
    if len(feature_names_list) <= 1: return feature_names_list

    logging.info("  Calculating association matrix for clustering...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_df = associations(original_df, nom_nom_assoc='cramer', compute_only=True)['corr'].fillna(0)

    logging.info("  Performing hierarchical clustering...")
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)
    if condensed_dist_mat.shape[0] == 0: return feature_names_list

    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')

    selected_representatives_list = []
    for i in range(1, len(np.unique(cluster_labels_arr)) + 1):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_indices: continue

        if len(cluster_indices) == 1:
            selected_representatives_list.append(feature_names_list[cluster_indices[0]])
        else:
            sum_abs_assoc = np.abs(assoc_df.iloc[cluster_indices, cluster_indices].values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc)
            selected_representatives_list.append(feature_names_list[cluster_indices[rep_local_idx]])

    return sorted(list(set(selected_representatives_list)))


def main():
    """Main function to run SHAP analysis on all high-performing model combinations."""
    logging.info(f"--- Starting Script: 7_shap.py (F1 > {PERFORMANCE_THRESHOLD}) ---")
    warnings.filterwarnings("ignore", category=UserWarning)

    # Step 1: Identify high-performing model combinations
    logging.info(
        f"Identifying combinations with Test F1 Weighted > {PERFORMANCE_THRESHOLD} from '{DETAILED_PERFORMANCE_CSV}'...")
    try:
        performance_df = pd.read_csv(DETAILED_PERFORMANCE_CSV)
        high_performing_combinations = performance_df[
            performance_df['Test F1 Weighted'] > PERFORMANCE_THRESHOLD].sort_values(by='Test F1 Weighted',
                                                                                    ascending=False)

        if high_performing_combinations.empty:
            logging.warning(
                f"No model combinations found with a Test F1 Weighted score > {PERFORMANCE_THRESHOLD}. Exiting.")
            sys.exit(0)

        logging.info(f"{len(high_performing_combinations)} high-performing combinations identified:")
        logging.info(high_performing_combinations[['Model', 'Feature Set Name', 'Test F1 Weighted']].to_string())
    except (FileNotFoundError, IndexError) as e:
        logging.error(f"FATAL: Could not identify top models from '{DETAILED_PERFORMANCE_CSV}'. Error: {e}")
        sys.exit(1)

    # Step 2: Set up all possible models and load all necessary data
    all_models = {
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1) if LGBM_AVAILABLE else None,
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss') if XGB_AVAILABLE else None,
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=2000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Ordinal LAD": mord.LAD() if MORD_AVAILABLE else None,
        "Ordinal Ridge": mord.OrdinalRidge() if MORD_AVAILABLE else None
    }

    X_train_orig = load_data(TRAIN_X_PATH, "original X_train")
    y_train = load_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_data(TEST_X_PATH, "original X_test")
    y_test = load_data(TEST_Y_PATH, "original y_test")

    X_train_sanitized = sanitize_feature_names(pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names(pd.DataFrame(X_test_orig)).reindex(columns=X_train_sanitized.columns,
                                                                                 fill_value=0)

    # Step 3: Loop through each high-performing combination and run analysis
    all_shap_values = {}
    all_test_samples = {}

    for index, combo_row in high_performing_combinations.iterrows():
        model_name = combo_row['Model']
        threshold = combo_row['Threshold Value']
        params_str = combo_row['Best Params']

        model_key = f"{model_name} (Thresh: {'Orig' if pd.isna(threshold) else threshold})"
        safe_model_key = re.sub(r'[^A-Za-z0-9_.-]+', '_', model_key)

        logging.info(f"\n===== Analyzing Combination: {model_key} =====")

        selected_features = get_selected_features_by_clustering(X_train_sanitized, threshold, CLUSTERING_LINKAGE_METHOD)
        X_train_selected = X_train_sanitized[selected_features]
        X_test_selected = X_test_sanitized[selected_features]

        model_template = all_models.get(model_name)
        if model_template is None: continue

        best_params = ast.literal_eval(params_str) if isinstance(params_str, str) and params_str != 'nan' else {}
        model_instance = model_template.set_params(**best_params)

        logging.info(f"  Retraining {model_name} on {len(selected_features)} features...")
        model_instance.fit(X_train_selected, y_train.ravel())

        test_sample = X_test_selected.sample(min(N_SHAP_SAMPLES, len(X_test_selected)), random_state=42)
        background_data = X_train_selected.sample(min(N_BACKGROUND_SAMPLES, len(X_train_selected)), random_state=42)

        logging.info("  Calculating SHAP values...")
        explainer = shap.Explainer(model_instance.predict_proba, background_data)
        shap_values = explainer(test_sample)

        all_shap_values[safe_model_key] = shap_values
        all_test_samples[safe_model_key] = test_sample

    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)
    unique_classes = sorted(np.unique(y_test))

    # Step 4: Create and log the composite bar chart
    logging.info("\n--- Generating Composite SHAP Bar Chart ---")
    all_importance_dfs = []
    for model_key, shap_values in all_shap_values.items():
        mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
        df = pd.DataFrame({
            'Feature': all_test_samples[model_key].columns,
            'Importance': mean_abs_shap,
            'Model': model_key
        })
        all_importance_dfs.append(df)

    combined_importance_df = pd.concat(all_importance_dfs, ignore_index=True)

    logging.info("\n--- Data for Composite Bar Chart ---")
    logging.info(f"\n{combined_importance_df.sort_values(by='Importance', ascending=False).to_string()}")

    top_features_order = combined_importance_df.groupby('Feature')['Importance'].max().nlargest(
        N_TOP_FEATURES_TO_PLOT).index
    plot_df = combined_importance_df[combined_importance_df['Feature'].isin(top_features_order)]

    plt.figure(figsize=(16, 12))
    sns.barplot(x='Importance', y='Feature', hue='Model', data=plot_df, palette='viridis', order=top_features_order)
    plt.title(f'Top {N_TOP_FEATURES_TO_PLOT} Feature Importances for Models with F1 > {PERFORMANCE_THRESHOLD}',
              fontsize=16)
    plt.xlabel('Mean Absolute SHAP Value (Average Impact on Model Output Magnitude)', fontsize=12)
    plt.ylabel('Feature Cluster', fontsize=12)
    plt.legend(title='Model (Threshold)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_RESULTS_DIR, 'shap_summary_bar_composite.png'))
    plt.close()
    logging.info("  Saved composite bar chart.")

    # Step 5: Create and log individual beeswarm plots
    logging.info("\n--- Generating SHAP Beeswarm Plots ---")
    for model_key, shap_values in all_shap_values.items():
        test_sample = all_test_samples[model_key]
        for i, target_class in enumerate(unique_classes):
            logging.info(f"\n--- Data for Beeswarm Plot: {model_key}, Class {target_class} ---")

            class_shap_values = shap_values[:, :, i]
            mean_abs_shap_class = np.abs(class_shap_values.values).mean(axis=0)
            class_importance_df = pd.DataFrame({
                'Feature': test_sample.columns,
                'Mean Absolute SHAP': mean_abs_shap_class
            }).sort_values(by='Mean Absolute SHAP', ascending=False)
            logging.info(f"\n{class_importance_df.head(N_TOP_FEATURES_TO_PLOT).to_string()}")

            plt.figure()
            shap.summary_plot(shap_values[:, :, i], test_sample, max_display=N_TOP_FEATURES_TO_PLOT, show=False)
            plt.title(f'SHAP Summary for Class {target_class}\n({model_key})', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_RESULTS_DIR, f'shap_beeswarm_class_{target_class}_{model_key}.png'))
            plt.close()
            logging.info(f"  Saved beeswarm plot for Class {target_class}, Model {model_key}.")

    logging.info(f"\nSHAP analysis complete. Plots saved to '{BASE_RESULTS_DIR}/'")
    logging.info(f"--- Finished Script: 7_shap.py ---")


if __name__ == "__main__":
    main()

