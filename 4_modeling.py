import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import logging
import sys
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import config

from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV


def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
                        force=True)


setup_logging()


def load_data(file_path, description="data"):
    logging.info(f"Loading {description} from {file_path}...")
    try:
        return joblib.load(file_path)
    except Exception as e:
        logging.error(f"Error loading {description}: {e}", exc_info=True)
        sys.exit(1)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()

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


def main():
    logging.info(f"--- Starting Script: 4_modeling.py (Regression Version) ---")
    os.makedirs(config.BASE_RESULTS_DIR, exist_ok=True)

    X_train = load_data(config.TRAIN_X_PATH, "training features")
    y_train = load_data(config.TRAIN_Y_PATH, "training target")
    X_test = load_data(config.TEST_X_PATH, "test features")
    y_test = load_data(config.Y_TEST_PATH, "test target")

    y_train_ravel = y_train.to_numpy().ravel()
    y_test_ravel = y_test.to_numpy().ravel()

    all_results = []
    best_estimators = {}

    for threshold in config.CLUSTERING_THRESHOLDS_TO_TEST:
        feature_set_label = f"Clustered (Thresh={threshold})" if threshold is not None else "Original Features"
        logging.info(f"\n===== PROCESSING FEATURE SET: {feature_set_label} =====")

        selected_features = get_selected_features_by_clustering(X_train, threshold, config.CLUSTERING_LINKAGE_METHOD)
        X_train_fs = X_train[selected_features]
        X_test_fs = X_test.reindex(columns=X_train_fs.columns, fill_value=0)

        for model_name, model_template in config.MODELS_TO_BENCHMARK.items():
            logging.info(f"  --- Benchmarking Model: {model_name} ---")
            param_grid = config.PARAM_GRIDS.get(model_name, {})
            kf_cv = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=config.RANDOM_STATE)

            grid_search = GridSearchCV(estimator=model_template, param_grid=param_grid,
                                       scoring=config.GRIDSEARCH_SCORING_METRIC,
                                       cv=kf_cv, n_jobs=-1, error_score='raise')

            try:
                grid_search.fit(X_train_fs, y_train_ravel)
                best_estimator = grid_search.best_estimator_
                combo_key = f"{model_name}_{feature_set_label}"
                best_estimators[combo_key] = best_estimator

                result_row = {
                    "Model": model_name, "Feature Set Name": feature_set_label,
                    "Number of Features": len(selected_features), "Threshold Value": threshold,
                    "Best Params": str(grid_search.best_params_)
                }

                y_pred_test = best_estimator.predict(X_test_fs)

                # Regression metrics
                result_row['Test R2 Score'] = r2_score(y_test_ravel, y_pred_test)
                result_row['Test MSE'] = mean_squared_error(y_test_ravel, y_pred_test)
                result_row['Test MAE'] = mean_absolute_error(y_test_ravel, y_pred_test)

                all_results.append(result_row)

            except Exception as e:
                logging.error(f"    ERROR running {model_name} for {feature_set_label}: {e}")
                continue

    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(config.DETAILED_RESULTS_CSV, index=False, float_format='%.6f')
    joblib.dump(best_estimators, config.BEST_ESTIMATORS_PATH)

    logging.info(f"\nComprehensive performance results saved to: {config.DETAILED_RESULTS_CSV}")
    logging.info(f"Saved dictionary of best estimators to: {config.BEST_ESTIMATORS_PATH}")

    # --- Print Console Report ---
    logging.info("\n--- Regression Model Performance Report ---")
    print(all_results_df.to_string())

    # --- Plot Model Comparison Bar Chart ---
    logging.info("\n--- Generating Model Comparison Plot ---")
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Test R2 Score', y='Model', data=all_results_df.sort_values('Test R2 Score', ascending=False),
                palette='viridis')
    plt.title('Model Comparison by R2 Score')
    plt.xlabel('R2 Score')
    plt.ylabel('Model')
    plt.grid(True)
    comparison_plot_filename = "model_comparison_r2_score.png"
    plt.savefig(os.path.join(config.BASE_RESULTS_DIR, comparison_plot_filename))
    plt.show()  # Display the plot
    plt.close()
    logging.info(f"  Saved model comparison plot to {comparison_plot_filename}")

    # --- Plot Actual vs. Predicted for Top 5 Models ---
    logging.info("\n--- Generating Actual vs. Predicted Plots for Top 5 Models ---")
    plt.style.use(config.VISUALIZATION['plot_style'])
    top_5 = all_results_df.sort_values(by='Test R2 Score', ascending=False).head(5)

    for _, row in top_5.iterrows():
        combo_key = f"{row['Model']}_{row['Feature Set Name']}"
        estimator = best_estimators[combo_key]

        selected_features_for_pred = get_selected_features_by_clustering(X_train, row['Threshold Value'],
                                                                         config.CLUSTERING_LINKAGE_METHOD)
        X_test_for_pred = X_test.reindex(columns=selected_features_for_pred, fill_value=0)
        y_pred = estimator.predict(X_test_for_pred)

        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=y_test_ravel, y=y_pred, alpha=0.5)
        plt.plot([min(y_test_ravel), max(y_test_ravel)], [min(y_test_ravel), max(y_test_ravel)], color='red',
                 linestyle='--')
        plt.title(f"Actual vs. Predicted for {combo_key}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.grid(True)

        plot_filename = f"actual_vs_predicted_{combo_key.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(config.BASE_RESULTS_DIR, plot_filename))
        plt.show()  # Display the plot
        plt.close()
        logging.info(f"  Saved plot for {combo_key}")

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()