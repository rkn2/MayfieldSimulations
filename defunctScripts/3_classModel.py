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

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, make_scorer
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Import Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Optional Imports
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
    # Append to the file created by previous scripts
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Call the setup function
setup_logging()


# --- Configuration ---
MODEL_TYPE_TO_RUN = 'both'
DATA_DIR = '../processed_ml_data'
CV_RESULTS_FILENAME = 'model_tuned_cv_results.csv'
CV_CHART_FILENAME = 'model_cv_comparison_chart.png'
TEST_RESULTS_FILENAME = 'model_test_set_results.csv'
TEST_CHART_FILENAME = 'model_test_set_comparison_chart.png'
PERFORM_FINAL_BEST_MODEL_EVALUATION = True
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
CV_RESULTS_CSV_PATH = os.path.join(DATA_DIR, CV_RESULTS_FILENAME)
CV_METRICS_CHART_SAVE_PATH = os.path.join(DATA_DIR, CV_CHART_FILENAME)
TEST_RESULTS_CSV_PATH = os.path.join(DATA_DIR, TEST_RESULTS_FILENAME)
TEST_METRICS_CHART_SAVE_PATH = os.path.join(DATA_DIR, TEST_CHART_FILENAME)
BEST_MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'best_tuned_classifier.pkl')
RANDOM_STATE = 42
N_SPLITS = 5
GRIDSEARCH_SCORING = 'f1_weighted'

METRICS_TO_EVALUATE = {
    'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted', 'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted', 'recall_weighted': 'recall_weighted',
}
METRICS_FOR_CV_CHART = [f"Mean CV {key.replace('_', ' ').title()}" for key in METRICS_TO_EVALUATE]
METRICS_FOR_TEST_CHART = [f"Test {key.replace('_', ' ').title()}" for key in METRICS_TO_EVALUATE]


# --- Model Definitions ---
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

MODELS_TO_TEST = {}
PARAM_GRIDS = {}
if MODEL_TYPE_TO_RUN.lower() == 'normal':
    logging.info("Configuration: Running NORMAL classification models.")
    MODELS_TO_TEST = NORMAL_MODELS_TO_TEST
    PARAM_GRIDS = NORMAL_PARAM_GRIDS
elif MODEL_TYPE_TO_RUN.lower() == 'ordinal':
    logging.info("Configuration: Running ORDINAL classification models.")
    if not MORD_AVAILABLE:
        logging.error("ERROR: 'mord' library not found, cannot run ordinal models. Exiting.")
        exit()
    MODELS_TO_TEST = ORDINAL_MODELS_TO_TEST
    PARAM_GRIDS = ORDINAL_PARAM_GRIDS
elif MODEL_TYPE_TO_RUN.lower() == 'both':
    logging.info("Configuration: Running BOTH normal and ordinal classification models.")
    MODELS_TO_TEST = NORMAL_MODELS_TO_TEST.copy()
    PARAM_GRIDS = NORMAL_PARAM_GRIDS.copy()
    if MORD_AVAILABLE:
        MODELS_TO_TEST.update(ORDINAL_MODELS_TO_TEST)
        PARAM_GRIDS.update(ORDINAL_PARAM_GRIDS)
    else:
        logging.warning("Warning: 'mord' not found, running only normal models.")
else:
    logging.error(f"Error: Invalid MODEL_TYPE_TO_RUN value: '{MODEL_TYPE_TO_RUN}'. Use 'normal', 'ordinal', or 'both'.")
    exit()

MODELS_TO_TUNE = {name: model for name, model in MODELS_TO_TEST.items() if name in PARAM_GRIDS}
PRIMARY_CV_METRIC = f"Mean CV {GRIDSEARCH_SCORING.replace('_', ' ').title()}"

def sanitize_feature_names(df):
    if not isinstance(df, pd.DataFrame):
        return df
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]
    df.columns = new_cols
    return df

def plot_model_comparison_metrics(results_df, metrics_to_plot, chart_title, save_path):
    try:
        # Log the data that will be plotted
        logging.info(f"Data for '{chart_title}' chart:\n{results_df.to_string()}")

        if 'Model' not in results_df.columns and results_df.index.name == 'Model':
            results_df_plot = results_df.reset_index()
        else:
            results_df_plot = results_df.copy()

        actual_metrics_to_plot = [m for m in metrics_to_plot if m in results_df_plot.columns]
        if not actual_metrics_to_plot:
            logging.warning(f"  No metrics found for plotting '{chart_title}'. Skipping chart.")
            return

        df_melted = results_df_plot.melt(id_vars='Model', value_vars=actual_metrics_to_plot, var_name='Metric', value_name='Score')
        df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
        df_melted.dropna(subset=['Score'], inplace=True)

        if df_melted.empty:
            logging.warning(f"  No valid numeric data to plot for '{chart_title}'.")
            return

        plt.figure(figsize=(15, 9))
        sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
        plt.title(chart_title, fontsize=18, pad=20)
        plt.xlabel('Model', fontsize=14, labelpad=15)
        plt.ylabel('Score', fontsize=14, labelpad=15)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        plt.legend(title='Metric', fontsize=11, title_fontsize=13, bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05)
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logging.info(f"  Successfully saved chart to {save_path}")
        plt.close()
    except Exception as e:
        logging.error(f"  Error generating chart '{chart_title}': {e}", exc_info=True)


# --- Main Script ---
logging.info(f"--- Starting Script: 3_classModel.py ---")
logging.info("Starting Model Benchmarking with Hyperparameter Tuning...")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.info("\nStep 1: Loading preprocessed data...")
try:
    X_train_orig = joblib.load(TRAIN_X_PATH)
    y_train = joblib.load(TRAIN_Y_PATH)
    X_test_orig = joblib.load(TEST_X_PATH)
    y_test = joblib.load(TEST_Y_PATH)
    logging.info(f"  Data loaded successfully from '{DATA_DIR}' directory.")

    X_train = sanitize_feature_names(X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else pd.DataFrame(X_train_orig))
    X_test = sanitize_feature_names(X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else pd.DataFrame(X_test_orig))

    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        logging.info("  Realigning X_test columns to match X_train...")
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

except Exception as e:
    logging.error(f"Error loading or processing data: {e}", exc_info=True)
    exit()

if isinstance(y_train, (pd.Series, pd.DataFrame)): y_train = y_train.values.ravel()
if isinstance(y_test, (pd.Series, pd.DataFrame)): y_test = y_test.values.ravel()

logging.info(f"\nStep 2: Running GridSearchCV for {len(MODELS_TO_TUNE)} models...")
tuned_cv_results_list = []
best_estimators = {}
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model_name, model_template in MODELS_TO_TUNE.items():
    logging.info(f"  Tuning model: {model_name}...")
    param_grid = PARAM_GRIDS.get(model_name, {})
    grid_search = GridSearchCV(estimator=model_template, param_grid=param_grid, scoring=GRIDSEARCH_SCORING, cv=kf, n_jobs=-1, verbose=0, error_score='raise')
    try:
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        best_estimator_from_gs = grid_search.best_estimator_
        best_estimators[model_name] = best_estimator_from_gs
        logging.info(f"    Best params found: {grid_search.best_params_}")

        cv_results_for_model = cross_validate(best_estimator_from_gs, X_train, y_train, cv=kf, scoring=METRICS_TO_EVALUATE, n_jobs=-1)

        result_row = {"Model": model_name, "Best Params": str(grid_search.best_params_), "GridSearch Time (s)": tuning_time, "Error": None}
        for metric_key, scorer_name in METRICS_TO_EVALUATE.items():
            result_row[f"Mean CV {metric_key.replace('_', ' ').title()}"] = np.mean(cv_results_for_model[f'test_{metric_key}'])
        tuned_cv_results_list.append(result_row)

    except Exception as e:
        logging.error(f"    ERROR: Model {model_name} failed. Error: {e}", exc_info=True)
        tuned_cv_results_list.append({"Model": model_name, "Error": str(e)})

logging.info("\nStep 3: Summarizing and saving Cross-Validation results...")
if not tuned_cv_results_list:
    logging.error("No models were successfully tuned. Exiting.")
    exit()

tuned_cv_results_df = pd.DataFrame(tuned_cv_results_list).sort_values(by=PRIMARY_CV_METRIC, ascending=True, na_position='last')
tuned_cv_results_df.to_csv(CV_RESULTS_CSV_PATH, index=False, float_format='%.6f')
logging.info(f"  CV results saved to {CV_RESULTS_CSV_PATH}")
plot_model_comparison_metrics(tuned_cv_results_df, METRICS_FOR_CV_CHART, 'Model CV Performance Comparison', CV_METRICS_CHART_SAVE_PATH)
logging.info("\n--- Tuned Cross-Validation Model Comparison ---")
logging.info(f"\n{tuned_cv_results_df.to_string(index=False, na_rep='N/A')}")

if PERFORM_FINAL_BEST_MODEL_EVALUATION and best_estimators:
    logging.info("\nStep 4: Evaluating all tuned models on the test set...")
    all_models_test_results_list = []
    for model_name, estimator in best_estimators.items():
        test_result_row = {"Model": model_name}
        try:
            y_pred = estimator.predict(X_test)
            for metric, scorer in METRICS_TO_EVALUATE.items():
                if 'f1' in metric:
                    score = f1_score(y_test, y_pred, average=metric.split('_')[-1], zero_division=0)
                elif metric == 'accuracy':
                    score = accuracy_score(y_test, y_pred)
                elif 'precision' in metric:
                    score = precision_score(y_test, y_pred, average=metric.split('_')[-1], zero_division=0)
                elif 'recall' in metric:
                    score = recall_score(y_test, y_pred, average=metric.split('_')[-1], zero_division=0)
                test_result_row[f"Test {metric.replace('_', ' ').title()}"] = score
            all_models_test_results_list.append(test_result_row)
        except Exception as e:
            logging.error(f"    Error evaluating {model_name} on test set: {e}", exc_info=True)
            all_models_test_results_list.append({"Model": model_name, "Error": str(e)})

    all_models_test_results_df = pd.DataFrame(all_models_test_results_list).sort_values(by=f"Test {GRIDSEARCH_SCORING.replace('_', ' ').title()}", ascending=False, na_position='last')
    all_models_test_results_df.to_csv(TEST_RESULTS_CSV_PATH, index=False, float_format='%.6f')
    logging.info(f"  Test set results saved to {TEST_RESULTS_CSV_PATH}")
    plot_model_comparison_metrics(all_models_test_results_df, METRICS_FOR_TEST_CHART, 'Model Test Set Performance', TEST_METRICS_CHART_SAVE_PATH)
    logging.info("\n--- All Models Test Set Performance ---")
    logging.info(f"\n{all_models_test_results_df.to_string(index=False, na_rep='N/A')}")

    logging.info("\nStep 5: Detailed report for the best model from CV...")
    best_cv_model_name = tuned_cv_results_df.iloc[0]['Model']
    best_estimator = best_estimators.get(best_cv_model_name)
    if best_estimator:
        logging.info(f"  Best model from CV: {best_cv_model_name}")
        y_pred_best = best_estimator.predict(X_test)
        report = classification_report(y_test, y_pred_best, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred_best)
        logging.info(f"\n  Classification Report (Test Set - Best CV Model):\n{report}")
        logging.info(f"\n  Confusion Matrix (Test Set - Best CV Model):\n{conf_matrix}")
        joblib.dump(best_estimator, BEST_MODEL_SAVE_PATH)
        logging.info(f"  Best model object saved to {BEST_MODEL_SAVE_PATH}")

logging.info("\nBenchmarking finished.")
logging.info(f"--- Finished Script: 3_classModel.py ---")