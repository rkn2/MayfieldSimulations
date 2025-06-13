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
import ast

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Scikit-learn ---
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, make_scorer
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# --- Model Definitions ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

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


# --- Logging Configuration ---
def setup_logging(log_file='pipeline.log'):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'), logging.StreamHandler(sys.stdout)],
        force=True
    )


setup_logging()

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'clustering_performance_results'
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
DETAILED_RESULTS_CSV = os.path.join(BASE_RESULTS_DIR, 'clustering_performance_detailed_results.csv')

RANDOM_STATE = 42
N_SPLITS_CV = 5
CLUSTERING_THRESHOLDS_TO_TEST = [None, 0.3, 0.8, 0.9]
CLUSTERING_LINKAGE_METHOD = 'average'
GRIDSEARCH_SCORING_METRIC = 'f1_weighted'

METRICS_TO_EVALUATE = {
    'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted', 'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted', 'recall_weighted': 'recall_weighted',
}

# --- Model and Hyperparameter Grid Definitions ---
MODELS_TO_BENCHMARK = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier()
}
if XGB_AVAILABLE: MODELS_TO_BENCHMARK["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss')
if LGBM_AVAILABLE: MODELS_TO_BENCHMARK["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)
if MORD_AVAILABLE:
    MODELS_TO_BENCHMARK.update({
        "Ordinal Logistic (AT)": mord.LogisticAT(), "Ordinal Ridge": mord.OrdinalRidge(), "Ordinal LAD": mord.LAD()
    })

PARAM_GRIDS = {
    "Logistic Regression": {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]},
    "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8, 10], 'min_samples_leaf': [5, 10, 15]},
    "Random Forest": {'n_estimators': [100, 150], 'max_depth': [6, 8, 10], 'min_samples_leaf': [5, 10]},
    "Gradient Boosting": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]},
    "Hist Gradient Boosting": {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [20, 31]},
    "KNN": {'n_neighbors': [5, 7, 9], 'weights': ['uniform', 'distance']}
}
if XGB_AVAILABLE: PARAM_GRIDS["XGBoost"] = {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]}
if LGBM_AVAILABLE: PARAM_GRIDS["LightGBM"] = {'n_estimators': [100], 'learning_rate': [0.05, 0.1],
                                              'num_leaves': [15, 25]}
if MORD_AVAILABLE:
    PARAM_GRIDS.update({
        "Ordinal Logistic (AT)": {'alpha': [0.1, 1.0, 10.0]}, "Ordinal Ridge": {'alpha': [0.1, 1.0, 10.0]},
        "Ordinal LAD": {'C': [0.1, 1.0, 10.0]}
    })


# --- Helper Functions ---
def load_data(file_path, description="data"):
    logging.info(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        logging.info(f"  Successfully loaded: {description}")
        return data
    except Exception as e:
        logging.error(f"Error loading {description} from {file_path}: {e}", exc_info=True)
        sys.exit(1)


def sanitize_feature_names(df):
    """Sanitizes DataFrame column names to match model's expectations."""
    if not isinstance(df, pd.DataFrame):
        return df

    # This regex replaces any character that is not a letter, number, or underscore with a single underscore.
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', str(col)) for col in df.columns]
    df.columns = new_cols
    return df


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    if distance_thresh is None or pd.isna(distance_thresh):
        return original_df.columns.tolist()
    feature_names_list = original_df.columns.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_df = associations(original_df, nom_nom_assoc='cramer', compute_only=True)['corr'].fillna(0)
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    condensed_dist_mat = squareform(distance_mat, checks=False)
    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    selected_representatives_list = []
    for i in range(1, np.max(cluster_labels_arr) + 1):
        cluster_indices = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_indices: continue
        if len(cluster_indices) == 1:
            selected_representatives_list.append(feature_names_list[cluster_indices[0]])
        else:
            sum_abs_assoc = np.abs(assoc_df.iloc[cluster_indices, cluster_indices].values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc)
            selected_representatives_list.append(feature_names_list[cluster_indices[rep_local_idx]])
    return sorted(list(set(selected_representatives_list)))


# --- Main Orchestration ---
def main():
    logging.info(f"--- Starting Script: 4_Clustering_and_Evaluation ---")
    warnings.filterwarnings("ignore", category=UserWarning)
    os.makedirs(BASE_RESULTS_DIR, exist_ok=True)

    X_train_orig = load_data(TRAIN_X_PATH, "original X_train")
    y_train = load_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_data(TEST_X_PATH, "original X_test")
    y_test = load_data(Y_TEST_PATH, "original y_test")

    # *** FIXED ***: Sanitize feature names right after loading
    X_train_df = sanitize_feature_names(pd.DataFrame(X_train_orig))
    X_test_df = sanitize_feature_names(pd.DataFrame(X_test_orig))

    y_train_ravel = y_train.ravel() if hasattr(y_train, 'ravel') else y_train
    y_test_ravel = y_test.ravel() if hasattr(y_test, 'ravel') else y_test

    all_results = []
    best_estimators_per_combo = {}

    for threshold in CLUSTERING_THRESHOLDS_TO_TEST:
        feature_set_label = f"Clustered (Thresh={threshold})" if threshold is not None else "Original Features"
        logging.info(f"\n===== PROCESSING FEATURE SET: {feature_set_label} =====")

        selected_features = get_selected_features_by_clustering(X_train_df, threshold, CLUSTERING_LINKAGE_METHOD)
        X_train_fs = X_train_df[selected_features]
        X_test_fs = X_test_df[selected_features]

        logging.info(f"  Number of features: {len(selected_features)}")

        for model_name, model_template in MODELS_TO_BENCHMARK.items():
            logging.info(f"  --- Benchmarking Model: {model_name} ---")

            param_grid = PARAM_GRIDS.get(model_name, {})
            kf_cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                estimator=model_template, param_grid=param_grid, scoring=GRIDSEARCH_SCORING_METRIC,
                cv=kf_cv, n_jobs=-1, verbose=0, error_score='raise'
            )

            try:
                start_time = time.time()
                grid_search.fit(X_train_fs, y_train_ravel)
                tuning_time = time.time() - start_time

                best_estimator = grid_search.best_estimator_
                combo_key = f"{model_name}_{feature_set_label}"
                best_estimators_per_combo[combo_key] = best_estimator

                result_row = {
                    "Model": model_name,
                    "Feature Set Name": feature_set_label,
                    "Number of Features": len(selected_features),
                    "Threshold Value": threshold,
                    "Best Params": str(grid_search.best_params_),
                    "GridSearch Time (s)": tuning_time
                }

                cv_results = cross_validate(best_estimator, X_train_fs, y_train_ravel, cv=kf_cv,
                                            scoring=METRICS_TO_EVALUATE, n_jobs=-1)
                for metric, scorer_name in METRICS_TO_EVALUATE.items():
                    result_row[f"Mean CV {metric.replace('_', ' ').title()}"] = np.mean(cv_results[f'test_{metric}'])

                y_pred_test = best_estimator.predict(X_test_fs)
                for metric, scorer_name in METRICS_TO_EVALUATE.items():
                    score = f1_score(y_test_ravel, y_pred_test, average=metric.split('_')[-1],
                                     zero_division=0) if 'f1' in metric else \
                        precision_score(y_test_ravel, y_pred_test, average=metric.split('_')[-1],
                                        zero_division=0) if 'precision' in metric else \
                            recall_score(y_test_ravel, y_pred_test, average=metric.split('_')[-1],
                                         zero_division=0) if 'recall' in metric else \
                                accuracy_score(y_test_ravel, y_pred_test)
                    result_row[f"Test {metric.replace('_', ' ').title()}"] = score

                all_results.append(result_row)
            except Exception as e:
                logging.error(f"    ERROR running {model_name} for {feature_set_label}: {e}")
                continue  # Skip to the next model if one fails

    # --- Step 4: Consolidate, Log, and Save All Results ---
    all_results_df = pd.DataFrame(all_results)

    logging.info("\n\n===== COMPREHENSIVE PERFORMANCE SUMMARY (ALL MODELS & FEATURE SETS) =====")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        logging.info(f"\n{all_results_df.to_string()}")

    all_results_df.to_csv(DETAILED_RESULTS_CSV, index=False, float_format='%.6f')
    logging.info(f"\nComprehensive performance results saved to: {DETAILED_RESULTS_CSV}")

    # --- Step 5: Identify and Generate Detailed Reports for Top 5 Models ---
    top_5_combinations = all_results_df.sort_values(by='Test F1 Weighted', ascending=False).head(5)
    logging.info("\n\n===== TOP 5 PERFORMING MODEL COMBINATIONS OVERALL =====")
    logging.info(f"\n{top_5_combinations[['Model', 'Feature Set Name', 'Test F1 Weighted']].to_string()}")

    for index, row in top_5_combinations.iterrows():
        combo_key = f"{row['Model']}_{row['Feature Set Name']}"
        estimator = best_estimators_per_combo.get(combo_key)

        if estimator is None:
            logging.warning(f"Could not find a stored estimator for {combo_key}. Skipping detailed report.")
            continue

        selected_features = get_selected_features_by_clustering(X_train_df, row['Threshold Value'],
                                                                CLUSTERING_LINKAGE_METHOD)
        X_test_fs = X_test_df[selected_features]

        y_pred = estimator.predict(X_test_fs)

        logging.info(f"\n--- Detailed Report for: {combo_key} ---")
        logging.info(f"Classification Report:\n{classification_report(y_test_ravel, y_pred, zero_division=0)}")

        cm = confusion_matrix(y_test_ravel, y_pred)
        logging.info(f"Confusion Matrix:\n{cm}")

        try:
            display_labels = sorted(np.unique(np.concatenate((y_test_ravel, y_pred))))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(ax=ax, cmap='Blues')
            plt.title(f"Confusion Matrix for {combo_key}")
            cm_filename = f"confusion_matrix_{re.sub(r'[^A-Za-z0-9_]+', '_', combo_key)}.png"
            plt.savefig(os.path.join(BASE_RESULTS_DIR, cm_filename))
            plt.close(fig)
            logging.info(f"  Confusion matrix plot saved to: {os.path.join(BASE_RESULTS_DIR, cm_filename)}")
        except Exception as e:
            logging.error(f"  Could not plot confusion matrix for {combo_key}. Error: {e}")

    logging.info("\n--- Script Finished ---")


if __name__ == '__main__':
    main()
