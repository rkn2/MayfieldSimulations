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
import shutil
from scipy.stats import pearsonr
from matplotlib.colors import ListedColormap

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

    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
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
RANDOM_STATE = 42


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


def plot_feature_correlation_heatmap(df, top_features, output_dir):
    """
    Calculates and plots a correlation heatmap for the specified top features.
    """
    logging.info("\n--- Generating Top Feature Correlation Heatmap ---")

    top_features_df = df[top_features]

    logging.info(f"Calculating associations for {len(top_features)} top features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_results = associations(top_features_df, nom_nom_assoc='cramer', compute_only=True, mark_columns=False)

    correlation_matrix = assoc_results['corr'].fillna(0)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=.5
    )
    plt.title('Cross-Correlation of Top SHAP Features', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    heatmap_filename = "top_features_correlation_heatmap.png"
    save_path = os.path.join(output_dir, heatmap_filename)
    try:
        plt.savefig(save_path)
        logging.info(f"Correlation heatmap saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save correlation heatmap. Error: {e}")

    plt.show()
    plt.close()


def plot_faceted_shap_summary(all_shap_values, all_test_samples, top_features, unique_classes, output_dir):
    """
    Creates a single figure with faceted beeswarm plots for each model and class.
    This version saves individual plots temporarily to ensure compatibility with all `shap` versions.
    """
    logging.info("\n--- Generating Faceted SHAP Beeswarm Plot for Consolidated Analysis ---")

    temp_plot_dir = os.path.join(output_dir, "temp_shap_plots")
    os.makedirs(temp_plot_dir, exist_ok=True)
    plot_paths = {}

    for model_key, shap_values in all_shap_values.items():
        plot_paths[model_key] = {}
        test_sample = all_test_samples[model_key]
        for class_idx, target_class in enumerate(unique_classes):
            plt.figure()
            shap.summary_plot(
                shap_values[:, :, class_idx],
                test_sample,
                max_display=len(top_features),
                show=False
            )
            temp_filename = f"temp_{model_key}_class_{target_class}.png"
            temp_filepath = os.path.join(temp_plot_dir, temp_filename)
            plt.savefig(temp_filepath, bbox_inches='tight')
            plt.close()
            plot_paths[model_key][target_class] = temp_filepath

    n_models = len(all_shap_values)
    n_classes = len(unique_classes)

    fig, axes = plt.subplots(n_models, n_classes, figsize=(n_classes * 8, n_models * 7), squeeze=False)

    for model_idx, model_key in enumerate(all_shap_values.keys()):
        for class_idx, target_class in enumerate(unique_classes):
            ax = axes[model_idx, class_idx]
            filepath = plot_paths[model_key].get(target_class)

            if filepath and os.path.exists(filepath):
                img = plt.imread(filepath)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Model: {model_key}\nClass: {target_class}", fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Plot not generated', ha='center', va='center')
                ax.axis('off')

    fig.suptitle("Faceted SHAP Summary: Feature Impact by Model and Damage Class", fontsize=20, y=1.03)
    plt.tight_layout(pad=3.0)

    faceted_plot_filename = "faceted_shap_summary.png"
    save_path = os.path.join(output_dir, faceted_plot_filename)
    try:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Faceted SHAP summary plot saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save faceted SHAP plot. Error: {e}")

    plt.show()
    plt.close(fig)

    try:
        shutil.rmtree(temp_plot_dir)
        logging.info(f"Successfully removed temporary plot directory: {temp_plot_dir}")
    except Exception as e:
        logging.error(f"Could not remove temporary plot directory: {temp_plot_dir}. Error: {e}")


# <<< --- START: REVISED SYNTHESIS HEATMAP FUNCTION --- >>>
def generate_feature_synthesis_report(all_shap_values, all_test_samples, top_features, unique_classes, output_dir):
    """
    Generates heatmaps synthesizing the impact of each top feature across all models.
    The color represents the direction of impact, and the annotation is the mean absolute SHAP value.
    """
    logging.info("\n\n" + "=" * 80)
    logging.info("--- Generating Feature Impact Synthesis Heatmaps ---")
    logging.info("=" * 80)

    # Prepare data for all classes first
    directional_data = {}
    magnitude_data = {}

    for model_key, test_sample in all_test_samples.items():
        directional_data[model_key] = {}
        magnitude_data[model_key] = {}
        shap_values_for_model = all_shap_values[model_key]

        for feature in top_features:
            if feature in test_sample.columns:
                feature_values = test_sample[feature]
                for class_idx, target_class in enumerate(unique_classes):
                    shap_values_for_class = shap_values_for_model[:, :, class_idx].values[:,
                                            test_sample.columns.get_loc(feature)]

                    # Store magnitude (mean absolute SHAP)
                    magnitude = np.abs(shap_values_for_class).mean()
                    magnitude_data[model_key][(feature, target_class)] = magnitude

                    # Store direction (-1 for Negative, 0 for Mixed, 1 for Positive)
                    try:
                        corr, _ = pearsonr(feature_values, shap_values_for_class)
                    except ValueError:
                        corr = np.nan

                    if pd.isna(corr) or abs(corr) < 0.1:
                        direction = 0  # Mixed/No Trend
                    elif corr > 0:
                        direction = 1  # Positive
                    else:
                        direction = -1  # Negative
                    directional_data[model_key][(feature, target_class)] = direction

    # Create and save one heatmap per class
    model_keys = list(all_test_samples.keys())

    for target_class in unique_classes:
        # Build the dataframes for this class's heatmap
        dir_df = pd.DataFrame(index=top_features, columns=model_keys, dtype=float)
        mag_df = pd.DataFrame(index=top_features, columns=model_keys, dtype=float)

        for feature in top_features:
            for model_key in model_keys:
                dir_df.loc[feature, model_key] = directional_data.get(model_key, {}).get((feature, target_class))
                mag_df.loc[feature, model_key] = magnitude_data.get(model_key, {}).get((feature, target_class))

        plt.figure(figsize=(16, 12))

        # Use the directional dataframe for coloring and the magnitude dataframe for annotations
        sns.heatmap(
            dir_df,  # This DataFrame contains -1, 0, 1 for direction
            annot=mag_df,  # This DataFrame contains the actual SHAP values for annotation
            fmt=".3f",
            cmap='coolwarm',  # A diverging colormap (Blue -> White -> Red)
            linewidths=.5,
            linecolor='black',
            cbar=False,
            center=0  # This ensures 0 (Mixed) is the center color (white)
        )

        plt.title(f"Feature Impact Synthesis for Class {target_class}", fontsize=18, pad=20)
        plt.xlabel("Model (Feature Set)", fontsize=12)
        plt.ylabel("Top 15 Features", fontsize=12)
        plt.xticks(rotation=45, ha='right')

        heatmap_filename = f"feature_synthesis_heatmap_class_{target_class}.png"
        save_path = os.path.join(output_dir, heatmap_filename)
        try:
            plt.savefig(save_path, bbox_inches='tight')
            logging.info(f"Successfully saved synthesis heatmap to: {save_path}")
        except Exception as e:
            logging.error(f"Failed to save synthesis heatmap for class {target_class}. Error: {e}")

        plt.show()
        plt.close()


# <<< --- END: REVISED SYNTHESIS HEATMAP FUNCTION --- >>>


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
        "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1) if LGB_AVAILABLE else None,
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss') if XGB_AVAILABLE else None,
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        # "KNN": KNeighborsClassifier(), # User commented out, noting sensitivity
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Ordinal LAD": mord.LAD() if MORD_AVAILABLE else None,
        "Ordinal Ridge": mord.OrdinalRidge() if MORD_AVAILABLE else None,
        "Ordinal Logistic (AT)": mord.LogisticAT() if MORD_AVAILABLE else None,
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

        if model_name == "KNN":
            logging.info(f"\nSkipping KNN model as it is disabled in the script.")
            continue

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

        tree_model_names = [
            "RandomForestClassifier", "HistGradientBoostingClassifier",
            "XGBClassifier", "LGBMClassifier", "DecisionTreeClassifier"
        ]

        if model_instance.__class__.__name__ in tree_model_names:
            logging.info("  Using shap.TreeExplainer for optimized tree-based model.")
            explainer = shap.TreeExplainer(model_instance, background_data)
            shap_values = explainer(test_sample, check_additivity=False)
        else:
            if model_instance.__class__.__name__ == "GradientBoostingClassifier":
                logging.info(
                    "  Model is GradientBoostingClassifier (multi-class); using model-agnostic shap.Explainer to avoid error.")
            else:
                logging.info("  Using shap.Explainer for model-agnostic explanation.")
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

    plot_feature_correlation_heatmap(X_train_sanitized, top_features_order, BASE_RESULTS_DIR)

    # Step 5: Create individual beeswarm plots (for reference/logging)
    logging.info("\n--- Generating Individual SHAP Beeswarm Plots (for logs and temp storage) ---")
    for model_key, shap_values in all_shap_values.items():
        test_sample = all_test_samples[model_key]
        for i, target_class in enumerate(unique_classes):
            plt.figure()
            shap.summary_plot(shap_values[:, :, i], test_sample, max_display=N_TOP_FEATURES_TO_PLOT, show=False)
            plt.title(f'SHAP Summary for Class {target_class}\n({model_key})', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_RESULTS_DIR, f'shap_beeswarm_class_{target_class}_{model_key}.png'))
            plt.close()

    plot_faceted_shap_summary(all_shap_values, all_test_samples, top_features_order, unique_classes, BASE_RESULTS_DIR)

    # Generate the synthesis report heatmaps
    generate_feature_synthesis_report(all_shap_values, all_test_samples, top_features_order, unique_classes,
                                      BASE_RESULTS_DIR)

    logging.info(f"\nSHAP analysis complete. Plots and reports saved to '{BASE_RESULTS_DIR}/'")
    logging.info(f"--- Finished Script: 7_shap.py ---")


if __name__ == "__main__":
    main()
