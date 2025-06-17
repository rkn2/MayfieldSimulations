import pandas as pd
import numpy as np
import os
import joblib
import warnings
import logging
import sys
import ast
import re
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import config

from scipy.stats import pearsonr


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
        logging.error(f"Error loading {description}: {e}", exc_info=True)
        sys.exit(1)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    """Recreates the clustered feature set for a given threshold."""
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()

    from dython.nominal import associations
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

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


# --- Visualization Functions ---

def plot_visualizations(all_shap_values, all_test_samples, output_dir):
    """Generates and saves all SHAP and feature correlation plots."""
    logging.info("\n--- Generating All Visualizations ---")
    plt.style.use(config.VISUALIZATION['plot_style'])

    # 1. Composite Bar Chart
    all_importance_dfs = []
    for model_key, shap_expl in all_shap_values.items():
        if len(shap_expl.values.shape) == 3:  # Multi-class output
            mean_abs_shap = np.abs(shap_expl.values).mean(axis=(0, 2))
        else:  # Single-class or other model output
            mean_abs_shap = np.abs(shap_expl.values).mean(axis=0)
        df = pd.DataFrame(
            {'Feature': all_test_samples[model_key].columns, 'Importance': mean_abs_shap, 'Model': model_key})
        all_importance_dfs.append(df)

    combined_importance_df = pd.concat(all_importance_dfs, ignore_index=True)
    top_features = combined_importance_df.groupby('Feature')['Importance'].max().nlargest(15).index

    plt.figure(figsize=(16, 12))
    sns.barplot(x='Importance', y='Feature', hue='Model',
                data=combined_importance_df[combined_importance_df['Feature'].isin(top_features)],
                palette=config.VISUALIZATION['main_palette'], order=top_features)
    plt.title('Top 15 Feature Importances (SHAP) Across High-Performing Models', fontsize=16)
    plt.xlabel('Mean Absolute SHAP Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_summary_bar_composite.png'))
    plt.close()
    logging.info("  Saved composite SHAP bar chart.")

    # 2. Beeswarm Plots
    for model_key, shap_expl in all_shap_values.items():
        if len(shap_expl.values.shape) == 3:  # Multi-class classification
            num_classes = shap_expl.values.shape[2]
            for i in range(num_classes):
                plt.figure()
                shap.summary_plot(shap_expl.values[:, :, i], all_test_samples[model_key], show=False, plot_type="dot")
                plt.title(f'SHAP Beeswarm for Class {i} ({model_key})', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shap_beeswarm_class_{i}_{model_key}.png'))
                plt.close()
        else:  # Regression or single output models
            plt.figure()
            shap.summary_plot(shap_expl.values, all_test_samples[model_key], show=False, plot_type="dot")
            plt.title(f'SHAP Beeswarm ({model_key})', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'shap_beeswarm_{model_key}.png'))
            plt.close()
    logging.info("  Saved individual SHAP beeswarm plots.")

    # 3. Synthesis Heatmaps
    plot_synthesis_heatmaps(all_shap_values, all_test_samples, top_features, output_dir)


def plot_synthesis_heatmaps(all_shap_values, all_test_samples, top_features, output_dir):
    """Generates heatmaps synthesizing feature impact across all models for each class."""
    logging.info("\n--- Generating Feature Impact Synthesis Heatmaps ---")

    # Determine the set of unique classes from all models
    unique_classes = set()
    for expl in all_shap_values.values():
        if len(expl.values.shape) == 3:
            for i in range(expl.values.shape[2]):
                unique_classes.add(i)
    if not unique_classes: unique_classes.add(0)  # For single-output models

    for class_idx in sorted(list(unique_classes)):
        directional_data = []
        for model_key, shap_expl in all_shap_values.items():
            model_impact = {}
            test_sample = all_test_samples[model_key]

            for feature in top_features:
                if feature not in test_sample.columns: continue

                if len(shap_expl.values.shape) == 3:
                    if class_idx >= shap_expl.values.shape[2]: continue
                    shap_vals_for_feature = shap_expl.values[:, test_sample.columns.get_loc(feature), class_idx]
                else:
                    if class_idx > 0: continue
                    shap_vals_for_feature = shap_expl.values[:, test_sample.columns.get_loc(feature)]

                magnitude = np.abs(shap_vals_for_feature).mean()
                corr, _ = pearsonr(test_sample[feature].values.astype(float), shap_vals_for_feature)
                direction = np.sign(corr) if not np.isnan(corr) and abs(corr) > 0.1 else 0
                model_impact[feature] = (direction, magnitude)

            directional_data.append({'model': model_key, **model_impact})

        df = pd.DataFrame(directional_data).set_index('model').T
        dir_df = df.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        mag_df = df.applymap(lambda x: x[1] if isinstance(x, tuple) else np.nan)

        plt.figure(figsize=(18, 14))
        sns.heatmap(dir_df, annot=mag_df, fmt=".3f", cmap=config.VISUALIZATION['diverging_palette'],
                    linewidths=.5, linecolor='black', cbar=False, center=0)

        plt.title(f"Feature Impact Synthesis for Class {class_idx}", fontsize=18)
        plt.ylabel("Top Features", fontsize=12)
        plt.xlabel("Model Combination", fontsize=12)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_dir, f'feature_synthesis_heatmap_class_{class_idx}.png'), bbox_inches='tight')
        plt.close()

    logging.info("  Saved synthesis heatmaps.")


# --- Main SHAP Analysis Script ---
def main():
    logging.info(f"--- Starting Script: 7_shap.py ---")
    os.makedirs(config.SHAP_RESULTS_DIR, exist_ok=True)

    X_train = load_data(config.TRAIN_X_PATH, "training features")
    y_train = load_data(config.TRAIN_Y_PATH, "training target")
    X_test = load_data(config.TEST_X_PATH, "test features")
    performance_df = pd.read_csv(config.DETAILED_RESULTS_CSV)

    high_performers = performance_df[performance_df['Test F1 Weighted'] > config.PERFORMANCE_THRESHOLD_FOR_PLOT]
    if high_performers.empty: return

    all_shap_values, all_test_samples = {}, {}
    tree_model_names = ["RandomForestClassifier", "HistGradientBoostingClassifier", "XGBClassifier", "LGBMClassifier",
                        "DecisionTreeClassifier"]
    mord_model_names = ["LogisticAT", "OrdinalRidge", "LAD"]

    for _, row in high_performers.iterrows():
        model_name = row['Model']
        combo_key = f"{model_name}_{row['Feature Set Name']}"
        logging.info(f"\n===== Analyzing SHAP for: {combo_key} =====")

        model_template = config.MODELS_TO_BENCHMARK.get(model_name)
        if model_template is None: continue

        best_params = ast.literal_eval(row['Best Params']) if isinstance(row['Best Params'], str) and row[
            'Best Params'] != 'nan' else {}
        model_instance = model_template.set_params(**best_params)

        selected_features = get_selected_features_by_clustering(X_train, row['Threshold Value'],
                                                                config.CLUSTERING_LINKAGE_METHOD)
        model_instance.fit(X_train[selected_features], y_train)

        test_sample = X_test[selected_features].sample(n=min(1000, len(X_test)), random_state=config.RANDOM_STATE)
        background_data = X_train[selected_features].sample(n=min(200, len(X_train)), random_state=config.RANDOM_STATE)

        logging.info("  Calculating SHAP values...")
        explainer_func = model_instance.predict
        if hasattr(model_instance, 'predict_proba'):
            explainer_func = model_instance.predict_proba

        if model_instance.__class__.__name__ in tree_model_names:
            explainer = shap.TreeExplainer(model_instance, background_data)
            shap_values = explainer(test_sample, check_additivity=False)
        else:
            explainer = shap.Explainer(explainer_func, background_data)
            shap_values = explainer(test_sample)

        all_shap_values[combo_key] = shap_values
        all_test_samples[combo_key] = test_sample

    if all_shap_values:
        plot_visualizations(all_shap_values, all_test_samples, config.SHAP_RESULTS_DIR)

    logging.info(f"--- Finished Script: 7_shap.py ---")


if __name__ == "__main__":
    main()