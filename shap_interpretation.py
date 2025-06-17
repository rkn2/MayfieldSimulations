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
    """Loads pickled data with logging and error handling."""
    logging.info(f"Loading {description} from {file_path}...")
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        logging.error(
            f"FATAL: {description} file not found at '{file_path}'. Please ensure the pipeline has been run up to this point.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading {description}: {e}", exc_info=True)
        sys.exit(1)


def get_original_feature_name(processed_name, cat_features):
    """
    Extracts the original feature name from a ColumnTransformer-generated name.
    """
    if processed_name.startswith('num__') or processed_name.startswith('remainder__'):
        return processed_name.split('__', 1)[1]
    if processed_name.startswith('cat__'):
        processed_suffix = processed_name.split('__', 1)[1]
        best_match = ''
        for cat_name in cat_features:
            if processed_suffix.startswith(cat_name + "_"):
                if len(cat_name) > len(best_match):
                    best_match = cat_name
        if best_match:
            return best_match
    return processed_name.split('__', 1)[-1]


def interpret_best_model():
    """
    Identifies the best model, interprets it with SHAP, prints a summary,
    and visualizes the relationships for the top features.
    """
    logging.info("--- Starting SHAP Interpretation and Visualization ---")

    # Define and create specific output directories
    shap_plot_dir = config.SHAP_RESULTS_DIR
    top_feature_plot_dir = 'top_feature_plots'  # New dedicated directory
    os.makedirs(shap_plot_dir, exist_ok=True)
    os.makedirs(top_feature_plot_dir, exist_ok=True)

    # 1. Load data and results
    X_train = load_data(config.TRAIN_X_PATH, "training features")
    X_test = load_data(config.TEST_X_PATH, "test features")
    best_estimators = load_data(config.BEST_ESTIMATORS_PATH, "best estimators dictionary")
    preprocessor = load_data(config.PREPROCESSOR_PATH, "preprocessor")

    try:
        performance_df = pd.read_csv(config.DETAILED_RESULTS_CSV)
        cleaned_df_for_viz = pd.read_csv(config.CLEANED_CSV_PATH)
    except FileNotFoundError as e:
        logging.error(f"FATAL: A required file was not found. Please run the full pipeline. Details: {e}")
        sys.exit(1)

    # 2. Identify the best model
    best_model_row = performance_df.loc[performance_df['Test R2 Score'].idxmax()]
    model_name = best_model_row['Model']
    feature_set_name = best_model_row['Feature Set Name']
    combo_key = f"{model_name}_{feature_set_name}"

    logging.info(f"\nIdentified Best Model: {model_name} with Feature Set: {feature_set_name}")
    logging.info(f"  - Test R2 Score: {best_model_row['Test R2 Score']:.4f}")
    logging.info(f"  - Test MAE: {best_model_row['Test MAE']:.4f}")

    # 3. Load the model and its specific feature set
    estimator = best_estimators.get(combo_key)
    if estimator is None:
        logging.error(f"Could not find estimator for key '{combo_key}' in the saved dictionary.")
        return

    X_train_fs = X_train[X_train.columns]
    X_test_fs = X_test.reindex(columns=X_train.columns, fill_value=0)

    # 4. Calculate SHAP values
    logging.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_test_fs)

    # 5. Print and Save SHAP Summary
    logging.info("\n--- SHAP Value Summary (Global Feature Importance) ---")
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_summary_df = pd.DataFrame({
        'Feature': X_test_fs.columns,
        'Mean Abs SHAP Value': mean_abs_shap
    })
    shap_summary_df = shap_summary_df.sort_values(by='Mean Abs SHAP Value', ascending=False)

    print(shap_summary_df.head(20).to_string())

    shap_summary_path = os.path.join(shap_plot_dir, 'shap_summary_details.csv')
    shap_summary_df.to_csv(shap_summary_path, index=False)
    logging.info(f"  Saved SHAP summary details to {shap_summary_path}")

    # 6. Generate and Save Core SHAP Visualizations
    logging.info("\n--- Generating Core SHAP Plots ---")
    plt.figure()
    shap.summary_plot(shap_values, X_test_fs, plot_type="bar", show=False)
    plt.title(f'Global Feature Importance for {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_plot_dir, 'shap_summary_bar.png'))
    plt.show()
    plt.close()
    logging.info("  Saved SHAP summary bar plot.")

    plt.figure()
    shap.summary_plot(shap_values, X_test_fs, show=False)
    plt.title(f'Detailed Feature Impact for {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_plot_dir, 'shap_beeswarm_plot.png'))
    plt.show()
    plt.close()
    logging.info("  Saved SHAP beeswarm plot.")

    # --- 7. Visualize Top Feature Relationships ---
    logging.info("\n--- Generating Plots for Top Features vs. Simulation Difference ---")
    original_cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
    shap_summary_df['Original_Feature'] = shap_summary_df['Feature'].apply(
        lambda x: get_original_feature_name(x, original_cat_features))
    top_original_features = shap_summary_df['Original_Feature'].unique()[:5]
    logging.info(f"\nVisualizing top 5 original features: {list(top_original_features)}")

    plt.style.use(config.VISUALIZATION['plot_style'])

    for feature in top_original_features:
        if feature not in cleaned_df_for_viz.columns:
            logging.warning(f"Warning: Feature '{feature}' not found in the cleaned dataframe. Skipping visualization.")
            continue

        plt.figure(figsize=(10, 6))

        if pd.api.types.is_numeric_dtype(cleaned_df_for_viz[feature]):
            sns.regplot(data=cleaned_df_for_viz, x=feature, y=config.TARGET_COLUMN,
                        line_kws={"color": "red"}, scatter_kws={'alpha': 0.5})
            plt.title(f'Simulation Difference vs. {feature}', fontsize=16)
        else:
            sns.stripplot(data=cleaned_df_for_viz, x=feature, y=config.TARGET_COLUMN, alpha=0.7, jitter=True)
            sns.boxplot(data=cleaned_df_for_viz, x=feature, y=config.TARGET_COLUMN, fliersize=0,
                        boxprops={'facecolor': 'None'})
            plt.title(f'Simulation Difference by {feature}', fontsize=16)
            plt.xticks(rotation=45, ha='right')

        plt.xlabel(feature, fontsize=12)
        plt.ylabel('Difference (Actual - Simulated)', fontsize=12)
        plt.tight_layout()

        # --- UPDATED: Save to the new dedicated directory ---
        plot_path = os.path.join(top_feature_plot_dir, f'relationship_{feature}.png')
        plt.savefig(plot_path)
        logging.info(f"  - Plot saved to: {plot_path}")
        plt.show()
        plt.close()

    logging.info("\n--- SHAP Interpretation Finished ---")


if __name__ == '__main__':
    interpret_best_model()
