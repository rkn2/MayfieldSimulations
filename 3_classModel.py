import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import re  # For feature name sanitization

# Updated metrics for classification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score, make_scorer
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Import CLASSIFICATION Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Optional: Try importing xgboost and lightgbm classifiers
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
CV_RESULTS_FILENAME = 'model_tuned_cv_results.csv'  # Renamed for clarity
CV_CHART_FILENAME = 'model_cv_comparison_chart.png'  # Renamed for clarity
TEST_RESULTS_FILENAME = 'model_test_set_results.csv'  # New for test set results
TEST_CHART_FILENAME = 'model_test_set_comparison_chart.png'  # New for test set chart

PERFORM_FINAL_BEST_MODEL_EVALUATION = True  # Set to True to run test set evaluations

TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

CV_RESULTS_CSV_PATH = os.path.join(DATA_DIR, CV_RESULTS_FILENAME)
CV_METRICS_CHART_SAVE_PATH = os.path.join(DATA_DIR, CV_CHART_FILENAME)
TEST_RESULTS_CSV_PATH = os.path.join(DATA_DIR, TEST_RESULTS_FILENAME)  # New
TEST_METRICS_CHART_SAVE_PATH = os.path.join(DATA_DIR, TEST_CHART_FILENAME)  # New
BEST_MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'best_tuned_classifier.pkl')

RANDOM_STATE = 42

# --- Cross-Validation & Tuning Settings ---
N_SPLITS = 5
GRIDSEARCH_SCORING = 'f1_weighted'  # Robust metric

# Define metrics for CV and Test set evaluation
# For test set, we'll derive column names like "Test Accuracy" from these keys
METRICS_TO_EVALUATE = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    # 'neg_log_loss': 'neg_log_loss', # Uncomment if models support predict_proba and it's relevant
    # 'roc_auc_ovr_weighted': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr')
}
# Metrics to display in the charts (must match keys in METRICS_TO_EVALUATE after prefixing)
METRICS_FOR_CV_CHART = [f"Mean CV {key.replace('_', ' ').title()}" for key in METRICS_TO_EVALUATE if
                        not key.startswith('neg_') and not key.startswith('roc_auc')]
METRICS_FOR_TEST_CHART = [f"Test {key.replace('_', ' ').title()}" for key in METRICS_TO_EVALUATE if
                          not key.startswith('neg_') and not key.startswith('roc_auc')]

MODELS_TO_TEST = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),  # Keep it simple for initial runs
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier()
}
if XGB_AVAILABLE:
    MODELS_TO_TEST["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False,
                                                  eval_metric='mlogloss')
if LGBM_AVAILABLE:
    MODELS_TO_TEST["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE,
                                                    verbosity=-1)  # verbosity=-1 for less output

PARAM_GRIDS = {
    "Logistic Regression": {
        'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1.0, 10.0], 'solver': ['liblinear']
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'], 'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5]
    },
    "Random Forest": {
        'n_estimators': [100, 150], 'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 3]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]
    },
    "Hist Gradient Boosting": {
        'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [31, 50], 'max_depth': [None, 10]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean']
    }
}
if XGB_AVAILABLE:
    PARAM_GRIDS["XGBoost"] = {
        'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5],
        'subsample': [0.7, 0.9], 'colsample_bytree': [0.7, 0.9]
    }
if LGBM_AVAILABLE:
    PARAM_GRIDS["LightGBM"] = {
        'n_estimators': [100, 150], 'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31], 'max_depth': [-1, 10]
    }

MODELS_TO_TUNE = {name: model for name, model in MODELS_TO_TEST.items() if name in PARAM_GRIDS}

PRIMARY_CV_METRIC = f"Mean CV {GRIDSEARCH_SCORING.replace('_', ' ').title()}"  # e.g. "Mean CV F1 Weighted"
HIGHER_IS_BETTER = True


# --- Helper Function for Sanitizing Feature Names ---
def sanitize_feature_names(df):
    """
    Sanitizes DataFrame column names to be compatible with models like LightGBM.
    Removes special JSON characters and ensures names are unique.
    """
    if not isinstance(df, pd.DataFrame):
        print("Warning: sanitize_feature_names received non-DataFrame input. Skipping.")
        return df

    original_cols = df.columns.tolist()
    # Replace any character not a letter, number, or underscore with an underscore
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_cols]
    # LightGBM specific: further replace characters like '[,]', '<' which might still cause issues
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]

    # Ensure uniqueness if sanitization creates duplicates
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


# --- Helper Function for Plotting ---
def plot_model_comparison_metrics(results_df, metrics_to_plot, chart_title, save_path):
    """
    Generates and saves a bar chart comparing specified metrics for different models.
    """
    try:
        # Ensure 'Model' is a column if it's an index
        if 'Model' not in results_df.columns and results_df.index.name == 'Model':
            results_df_plot = results_df.reset_index()
        else:
            results_df_plot = results_df.copy()

        # Filter out metrics not present in the DataFrame to avoid errors
        actual_metrics_to_plot = [m for m in metrics_to_plot if m in results_df_plot.columns]
        if not actual_metrics_to_plot:
            print(
                f"  No metrics from the provided list found in results_df columns for plotting '{chart_title}'. Skipping chart.")
            return

        df_melted = results_df_plot.melt(id_vars='Model', value_vars=actual_metrics_to_plot,
                                         var_name='Metric', value_name='Score')

        df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
        df_melted.dropna(subset=['Score'], inplace=True)

        if df_melted.empty:
            print(f"  No valid numeric data to plot for '{chart_title}' after attempting to convert scores.")
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
        plt.ylim(0, 1.05)  # Set y-axis limit
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout for legend

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"  Successfully saved model comparison chart to {save_path}")
        plt.close()
    except Exception as e:
        print(f"  Error generating or saving model comparison chart '{chart_title}': {e}")
        import traceback
        traceback.print_exc()


# --- Main Script ---
print("Starting Ordinal Classification Model Benchmarking with Hyperparameter Tuning...")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

print("\nStep 1: Loading preprocessed data...")
try:
    X_train_orig = joblib.load(TRAIN_X_PATH)
    y_train = joblib.load(TRAIN_Y_PATH)
    X_test_orig = joblib.load(TEST_X_PATH)
    y_test = joblib.load(TEST_Y_PATH)

    if not isinstance(X_train_orig, pd.DataFrame) or not isinstance(X_test_orig, pd.DataFrame):
        # Attempt to convert if they are numpy arrays with feature names from preprocessor
        # This scenario is less likely if preprocessor saved them as DataFrames
        print(
            "Warning: X_train_processed.pkl or X_test_processed.pkl were not DataFrames. Attempting conversion if possible.")
        # This part would need access to feature_names_out from the preprocessing script,
        # or assume they are just numpy arrays without column names if that's how they were saved.
        # For now, we'll proceed assuming they should be DataFrames.
        # If they are numpy, sanitization might not be directly applicable or needed in the same way.
        pass  # Continue, sanitization will check instance type

    print(f"  Data loaded successfully from '{DATA_DIR}' directory.")
    print(f"  Original X_train shape: {X_train_orig.shape}, y_train shape: {y_train.shape}")
    print(f"  Original X_test shape: {X_test_orig.shape}, y_test shape: {y_test.shape}")

    # --- Sanitize feature names ---
    print("  Sanitizing feature names for X_train and X_test...")
    X_train = sanitize_feature_names(X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else X_train_orig)
    X_test = sanitize_feature_names(X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else X_test_orig)

    if isinstance(X_train, pd.DataFrame) and isinstance(X_test, pd.DataFrame):
        print(f"  Sanitized X_train columns (first 5): {X_train.columns[:5].tolist()}...")
        if not X_train.columns.equals(X_test.columns):
            print("  WARNING: X_train and X_test columns do not match after sanitization. This can cause issues.")
            # Attempt to align columns based on X_train - models are trained on X_train's features
            common_cols = X_train.columns.intersection(X_test.columns)
            missing_in_test = X_train.columns.difference(X_test.columns)
            missing_in_train = X_test.columns.difference(X_train.columns)

            if not missing_in_test.empty:
                print(
                    f"    Columns in X_train but not X_test (will be added to X_test with 0s): {missing_in_test.tolist()}")
                for col in missing_in_test:
                    X_test[col] = 0
            if not missing_in_train.empty:
                print(
                    f"    Columns in X_test but not X_train (will be dropped from X_test): {missing_in_train.tolist()}")

            X_test = X_test[X_train.columns]  # Reorder and select X_test columns to match X_train
            print(f"    X_test columns realigned with X_train. New X_test shape: {X_test.shape}")

    print(
        f"  Unique y_train labels: {np.unique(y_train)} (Distribution: {pd.Series(y_train).value_counts(normalize=True).sort_index().to_dict()})")
    print(
        f"  Unique y_test labels: {np.unique(y_test)} (Distribution: {pd.Series(y_test).value_counts(normalize=True).sort_index().to_dict()})")

except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}. Ensure preprocessing script ran successfully and paths are correct.")
    exit()
except Exception as e:
    print(f"Error loading or sanitizing data: {e}")
    import traceback

    traceback.print_exc()
    exit()

if isinstance(y_train, (pd.Series, pd.DataFrame)):
    y_train = y_train.values.ravel()
if isinstance(y_test, (pd.Series, pd.DataFrame)):
    y_test = y_test.values.ravel()

print(f"\nStep 2: Running GridSearchCV with {N_SPLITS}-Fold CV for each model...")
tuned_cv_results_list = []
best_estimators = {}  # To store the best estimator for each model type
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model_name, model_template in MODELS_TO_TUNE.items():
    print(f"  Tuning model: {model_name}...")
    param_grid = PARAM_GRIDS.get(model_name, {})

    # Determine which metrics can be used for CV for this specific model
    current_cv_scoring_report_for_model = METRICS_TO_EVALUATE.copy()
    has_predict_proba = hasattr(model_template, 'predict_proba')

    metrics_to_remove_from_cv = []
    if not has_predict_proba:
        if 'roc_auc_ovr_weighted' in current_cv_scoring_report_for_model:
            metrics_to_remove_from_cv.append('roc_auc_ovr_weighted')
        if 'neg_log_loss' in current_cv_scoring_report_for_model:
            metrics_to_remove_from_cv.append('neg_log_loss')

    for metric_key in metrics_to_remove_from_cv:
        del current_cv_scoring_report_for_model[metric_key]
        # print(f"    Model {model_name} does not support predict_proba. Removing {metric_key} from its CV report.")

    # Determine GridSearchCV scoring, fallback if primary choice needs predict_proba and model doesn't have it
    current_gridsearch_scoring_for_model = GRIDSEARCH_SCORING
    if ('roc_auc' in GRIDSEARCH_SCORING or 'log_loss' in GRIDSEARCH_SCORING) and not has_predict_proba:
        print(f"    Model {model_name} does not support predict_proba. GridSearch scoring will use 'f1_weighted'.")
        current_gridsearch_scoring_for_model = 'f1_weighted'

    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=model_template, param_grid=param_grid, scoring=current_gridsearch_scoring_for_model,
        cv=kf, n_jobs=-1, verbose=0, error_score='raise'  # error_score='raise' for debugging
    )
    try:
        grid_search.fit(X_train, y_train)
        tuning_time = time.time() - start_time
        best_params = grid_search.best_params_
        best_estimator_from_gs = grid_search.best_estimator_  # Already refit on full X_train if refit=True (default)
        best_estimators[model_name] = best_estimator_from_gs  # Store the fitted best estimator

        print(f"    Best params found: {best_params}")
        print(f"    Running final CV on best '{model_name}' estimator to gather all defined metrics...")
        # Use the best estimator found by GridSearchCV for this final CV metric gathering
        cv_results_for_model = cross_validate(
            best_estimator_from_gs, X_train, y_train, cv=kf, scoring=current_cv_scoring_report_for_model, n_jobs=-1,
            error_score='raise'
        )
    except Exception as e:
        print(f"    ERROR: Model {model_name} failed during GridSearchCV or subsequent CV. Error: {e}")
        import traceback

        traceback.print_exc()
        result_row = {"Model": model_name, "Best Params": "N/A", "GridSearch Time (s)": np.nan, "Error": str(e)}
        for metric_key in METRICS_TO_EVALUATE.keys():  # Ensure all potential columns are present
            result_row[f"Mean CV {metric_key.replace('_', ' ').title()}"] = np.nan
        tuned_cv_results_list.append(result_row)
        continue

    # Populate results for this model
    result_row = {"Model": model_name, "Best Params": str(best_params), "GridSearch Time (s)": round(tuning_time, 4),
                  "Error": None}
    print_metrics_log = []
    for key_metric_std_name, scorer_name_in_sklearn in METRICS_TO_EVALUATE.items():
        # Construct the key as used by cross_validate (e.g., 'test_accuracy', 'test_f1_weighted')
        cv_result_key_for_sklearn = f'test_{key_metric_std_name}'
        # Standardized column name for our results DataFrame
        result_col_name_for_df = f"Mean CV {key_metric_std_name.replace('_', ' ').title()}"

        if key_metric_std_name in current_cv_scoring_report_for_model:  # Metric was applicable and attempted for this model
            metric_values = cv_results_for_model.get(cv_result_key_for_sklearn, [])
            if len(metric_values) > 0:
                mean_metric = np.mean(metric_values)
                if scorer_name_in_sklearn.startswith('neg_'):  # Handle negative scorers
                    mean_metric = -mean_metric
                result_row[result_col_name_for_df] = mean_metric
                print_metrics_log.append(f"{result_col_name_for_df}: {mean_metric:.4f}")
            else:  # Metric was in current_cv_scoring_report_for_model but not in cv_results_for_model (should not happen with error_score='raise')
                result_row[result_col_name_for_df] = np.nan
                print_metrics_log.append(f"{result_col_name_for_df}: N/A (not found in cv_results)")
        else:  # Metric was not applicable (e.g., ROC AUC for model without predict_proba)
            result_row[result_col_name_for_df] = np.nan
            # print_metrics_log.append(f"{result_col_name_for_df}: N/A (skipped for this model)") # Optional logging

    tuned_cv_results_list.append(result_row)
    print(f"    Finished CV for {model_name} in {tuning_time:.2f} seconds. Metrics: {', '.join(print_metrics_log)}")

print("\nStep 3: Summarizing Tuned Cross-Validation results...")
if not tuned_cv_results_list:
    print("No models were successfully tuned or evaluated with CV.")
    # Decide if script should exit or try to proceed if PERFORM_FINAL_BEST_MODEL_EVALUATION is True
    # For now, assume if CV fails, we might not have best_estimators to evaluate on test set.
    if not PERFORM_FINAL_BEST_MODEL_EVALUATION:
        exit()
    else:  # Create an empty DataFrame to avoid errors later if we proceed
        tuned_cv_results_df = pd.DataFrame(
            columns=['Model'] + [f"Mean CV {k.replace('_', ' ').title()}" for k in METRICS_TO_EVALUATE.keys()])
else:
    tuned_cv_results_df = pd.DataFrame(tuned_cv_results_list)

# Ensure all expected metric columns exist from METRICS_TO_EVALUATE, fill with NaN if a model failed entirely
expected_cv_metric_cols = [f"Mean CV {key.replace('_', ' ').title()}" for key in METRICS_TO_EVALUATE.keys()]
for col in expected_cv_metric_cols:
    if col not in tuned_cv_results_df.columns:
        tuned_cv_results_df[col] = np.nan

# Fallback for PRIMARY_CV_METRIC if not found or all NaN
if PRIMARY_CV_METRIC not in tuned_cv_results_df.columns or tuned_cv_results_df[PRIMARY_CV_METRIC].isnull().all():
    print(f"Warning: Primary CV metric '{PRIMARY_CV_METRIC}' not found or all its values are NaN.")
    fallback_options = [f"Mean CV {m.replace('_', ' ').title()}" for m in ['f1_weighted', 'f1_macro', 'accuracy']]
    found_fallback = False
    for fallback_metric in fallback_options:
        if fallback_metric in tuned_cv_results_df.columns and not tuned_cv_results_df[fallback_metric].isnull().all():
            PRIMARY_CV_METRIC = fallback_metric
            print(f"Falling back to '{PRIMARY_CV_METRIC}' as primary CV metric.")
            found_fallback = True
            break
    if not found_fallback:
        print("Could not find a suitable fallback primary CV metric. Sorting may be affected or use model name.")
        if not tuned_cv_results_df.empty:
            tuned_cv_results_df_sorted = tuned_cv_results_df.sort_values(by="Model").copy()
        else:
            tuned_cv_results_df_sorted = tuned_cv_results_df.copy()
        PRIMARY_CV_METRIC = "Model"  # Placeholder
    else:
        tuned_cv_results_df_sorted = tuned_cv_results_df.sort_values(
            by=PRIMARY_CV_METRIC, ascending=not HIGHER_IS_BETTER, na_position='last'
        ).copy()
else:
    tuned_cv_results_df_sorted = tuned_cv_results_df.sort_values(
        by=PRIMARY_CV_METRIC, ascending=not HIGHER_IS_BETTER, na_position='last'
    ).copy()

print(f"\nStep 3.1: Generating model CV performance comparison chart (Sorted by: {PRIMARY_CV_METRIC})...")
if not tuned_cv_results_df_sorted.empty:
    plot_model_comparison_metrics(tuned_cv_results_df_sorted, METRICS_FOR_CV_CHART,
                                  'Model CV Performance Comparison', CV_METRICS_CHART_SAVE_PATH)
else:
    print("  Skipping CV chart generation as no results were generated.")

print(f"\nStep 3.2: Saving tuned CV results summary to {CV_RESULTS_CSV_PATH}...")
if not tuned_cv_results_df_sorted.empty:
    try:
        target_dir = os.path.dirname(CV_RESULTS_CSV_PATH)
        if target_dir: os.makedirs(target_dir, exist_ok=True)
        tuned_cv_results_df_sorted.to_csv(CV_RESULTS_CSV_PATH, index=False, float_format='%.6f')
        print(f"  Successfully saved tuned CV results to {CV_RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"Error: Could not save tuned CV results to CSV. Error: {e}")
else:
    print("  Skipping saving CV results CSV as no results were generated.")

print("\n--- Tuned Cross-Validation Model Comparison (Display) ---")
if not tuned_cv_results_df_sorted.empty:
    # Display logic from previous version
    display_df = tuned_cv_results_df_sorted.copy()
    for col in display_df.select_dtypes(include=np.number).columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    print(display_df.to_string(index=False, na_rep="N/A"))
else:
    print("No tuned CV results to display.")

# --- Steps 4 & 5: Test Set Evaluation for ALL models and Detailed Report for Best CV Model ---
if PERFORM_FINAL_BEST_MODEL_EVALUATION:
    if not best_estimators:
        print("\nNo best estimators found from CV tuning. Skipping all test set evaluations.")
    else:
        print("\nStep 4: Evaluating all tuned models on the hold-out test set...")
        all_models_test_results_list = []

        for model_name, estimator in best_estimators.items():
            print(f"  Evaluating {model_name} on test set...")
            test_result_row = {"Model": model_name, "Error": None}
            try:
                y_pred_test = estimator.predict(X_test)
                y_pred_proba_test = None
                if hasattr(estimator, "predict_proba"):
                    y_pred_proba_test = estimator.predict_proba(X_test)

                for metric_key, sk_scorer_name in METRICS_TO_EVALUATE.items():
                    test_metric_col_name = f"Test {metric_key.replace('_', ' ').title()}"
                    try:
                        if metric_key == 'accuracy':
                            test_result_row[test_metric_col_name] = accuracy_score(y_test, y_pred_test)
                        elif metric_key == 'f1_weighted':
                            test_result_row[test_metric_col_name] = f1_score(y_test, y_pred_test, average='weighted',
                                                                             zero_division=0)
                        elif metric_key == 'f1_macro':
                            test_result_row[test_metric_col_name] = f1_score(y_test, y_pred_test, average='macro',
                                                                             zero_division=0)
                        elif metric_key == 'precision_weighted':
                            test_result_row[test_metric_col_name] = precision_score(y_test, y_pred_test,
                                                                                    average='weighted', zero_division=0)
                        elif metric_key == 'recall_weighted':
                            test_result_row[test_metric_col_name] = recall_score(y_test, y_pred_test,
                                                                                 average='weighted', zero_division=0)
                        elif metric_key == 'neg_log_loss' and y_pred_proba_test is not None:
                            labels_for_metric = estimator.classes_ if hasattr(estimator, 'classes_') else sorted(
                                np.unique(y_test))
                            if y_pred_proba_test.shape[1] == len(labels_for_metric):
                                test_result_row[test_metric_col_name.replace("Neg ", "")] = log_loss(y_test,
                                                                                                     y_pred_proba_test,
                                                                                                     labels=labels_for_metric)  # Store positive log loss
                            else:
                                test_result_row[test_metric_col_name.replace("Neg ", "")] = np.nan
                        elif metric_key == 'roc_auc_ovr_weighted' and y_pred_proba_test is not None:
                            labels_for_metric = estimator.classes_ if hasattr(estimator, 'classes_') else sorted(
                                np.unique(y_test))
                            if len(labels_for_metric) > 1 and y_pred_proba_test.shape[1] == len(labels_for_metric):
                                test_result_row[test_metric_col_name] = roc_auc_score(y_test, y_pred_proba_test,
                                                                                      average='weighted',
                                                                                      multi_class='ovr',
                                                                                      labels=labels_for_metric)
                            else:
                                test_result_row[test_metric_col_name] = np.nan
                        # else:
                        #     test_result_row[test_metric_col_name] = np.nan # Metric not handled or proba not available
                    except Exception as e_metric:
                        print(f"    Could not calculate test metric {metric_key} for {model_name}: {e_metric}")
                        test_result_row[test_metric_col_name] = np.nan
                all_models_test_results_list.append(test_result_row)
            except Exception as e_eval:
                print(f"    ERROR: Failed to evaluate {model_name} on test set. Error: {e_eval}")
                error_row = {"Model": model_name, "Error": str(e_eval)}
                for metric_key in METRICS_TO_EVALUATE.keys():
                    error_row[f"Test {metric_key.replace('_', ' ').title()}"] = np.nan
                all_models_test_results_list.append(error_row)

        all_models_test_results_df = pd.DataFrame(all_models_test_results_list)

        # Sort test results, e.g., by Test F1 Weighted
        primary_test_metric_col = f"Test {GRIDSEARCH_SCORING.replace('_', ' ').title()}"
        if primary_test_metric_col not in all_models_test_results_df.columns:  # Fallback if primary scoring not in test metrics
            primary_test_metric_col = "Test F1 Weighted"  # Default fallback
        if primary_test_metric_col in all_models_test_results_df.columns:
            all_models_test_results_df_sorted = all_models_test_results_df.sort_values(
                by=primary_test_metric_col, ascending=not HIGHER_IS_BETTER, na_position='last'
            ).copy()
        else:
            all_models_test_results_df_sorted = all_models_test_results_df.sort_values(by="Model").copy()

        print(
            f"\nStep 4.1: Generating model test set performance comparison chart (Sorted by {primary_test_metric_col})...")
        if not all_models_test_results_df_sorted.empty:
            plot_model_comparison_metrics(all_models_test_results_df_sorted, METRICS_FOR_TEST_CHART,
                                          'Model Test Set Performance Comparison', TEST_METRICS_CHART_SAVE_PATH)
        else:
            print("  Skipping test set chart generation as no test results were generated.")

        print(f"\nStep 4.2: Saving all models' test set results to {TEST_RESULTS_CSV_PATH}...")
        if not all_models_test_results_df_sorted.empty:
            try:
                target_dir = os.path.dirname(TEST_RESULTS_CSV_PATH)
                if target_dir: os.makedirs(target_dir, exist_ok=True)
                all_models_test_results_df_sorted.to_csv(TEST_RESULTS_CSV_PATH, index=False, float_format='%.6f')
                print(f"  Successfully saved test set results to {TEST_RESULTS_CSV_PATH}")
            except Exception as e:
                print(f"Error: Could not save test set results to CSV. Error: {e}")
        else:
            print("  Skipping saving test set results CSV as no results were generated.")

        print("\n--- All Models Test Set Performance (Display) ---")
        if not all_models_test_results_df_sorted.empty:
            display_test_df = all_models_test_results_df_sorted.copy()
            for col in display_test_df.select_dtypes(include=np.number).columns:
                display_test_df[col] = display_test_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            print(display_test_df.to_string(index=False, na_rep="N/A"))
        else:
            print("No test set results to display for all models.")

        # --- Step 5: Detailed evaluation of the single BEST model (from CV) on the test set ---
        print("\nStep 5: Identifying best tuned model based on Cross-Validation for detailed report...")
        best_cv_model_name = None
        best_cv_final_estimator = None

        # Use tuned_cv_results_df_sorted which is already sorted by PRIMARY_CV_METRIC
        if not tuned_cv_results_df_sorted.empty and PRIMARY_CV_METRIC != "Model":
            # Filter out rows with errors or NaN in primary CV metric
            valid_cv_results = tuned_cv_results_df_sorted.dropna(subset=[PRIMARY_CV_METRIC])
            if 'Error' in valid_cv_results.columns:
                valid_cv_results = valid_cv_results[valid_cv_results['Error'].isna()]

            if not valid_cv_results.empty:
                best_cv_model_row = valid_cv_results.iloc[0]
                best_cv_model_name = best_cv_model_row['Model']
                if best_cv_model_name in best_estimators:
                    best_cv_final_estimator = best_estimators[best_cv_model_name]
                    print(f"\nBest model based on CV ({PRIMARY_CV_METRIC}):")
                    print(f"  Model: {best_cv_model_row['Model']}")
                    print(f"  CV {PRIMARY_CV_METRIC}: {best_cv_model_row[PRIMARY_CV_METRIC]:.4f}" if pd.notna(
                        best_cv_model_row[PRIMARY_CV_METRIC]) else "N/A")
                    print(f"  Best Params: {best_cv_model_row['Best Params']}")
                else:
                    print(f"Error: Best CV model '{best_cv_model_name}' not found in stored estimators dictionary.")
                    best_cv_model_name = None  # Reset
            else:
                print("Could not determine the best model from CV results (no valid results after filtering).")
        else:
            print("Could not determine the best model from CV (no CV results or primary metric invalid).")

        if best_cv_model_name and best_cv_final_estimator:
            print(f"\n--- Detailed Test Set Performance for Best CV Model: {best_cv_model_name} ---")
            try:
                y_pred_test_best_cv_model = best_cv_final_estimator.predict(X_test)
                y_pred_proba_test_best_cv_model = None
                if hasattr(best_cv_final_estimator, "predict_proba"):
                    y_pred_proba_test_best_cv_model = best_cv_final_estimator.predict_proba(X_test)

                print(f"  Model: {best_cv_model_name}")
                print(
                    f"  Params Used: {best_cv_final_estimator.get_params() if hasattr(best_cv_final_estimator, 'get_params') else 'N/A'}")

                # Re-calculate and print key metrics for this specific model for clarity
                print(f"  Test Accuracy:        {accuracy_score(y_test, y_pred_test_best_cv_model):.4f}")
                print(
                    f"  Test F1 Score (Weighted): {f1_score(y_test, y_pred_test_best_cv_model, average='weighted', zero_division=0):.4f}")
                print(
                    f"  Test F1 Score (Macro):    {f1_score(y_test, y_pred_test_best_cv_model, average='macro', zero_division=0):.4f}")

                if y_pred_proba_test_best_cv_model is not None and len(np.unique(y_test)) > 1:
                    try:
                        labels_for_roc = best_cv_final_estimator.classes_ if hasattr(best_cv_final_estimator,
                                                                                     'classes_') else sorted(
                            np.unique(y_test))
                        if len(labels_for_roc) > 1 and y_pred_proba_test_best_cv_model.shape[1] == len(labels_for_roc):
                            print(
                                f"  Test ROC AUC (OvR Wtd): {roc_auc_score(y_test, y_pred_proba_test_best_cv_model, average='weighted', multi_class='ovr', labels=labels_for_roc):.4f}")
                            print(
                                f"  Test Log Loss:          {log_loss(y_test, y_pred_proba_test_best_cv_model, labels=labels_for_roc):.4f}")
                        else:
                            print(
                                "  Skipping ROC AUC/Log Loss for best CV model on test set: class/probability shape mismatch.")
                    except Exception as e_roc_best:
                        print(
                            f"    Could not calculate ROC AUC or Log Loss for best CV model on test set: {e_roc_best}")

                print("\n  Classification Report (Test Set - Best CV Model):\n",
                      classification_report(y_test, y_pred_test_best_cv_model, zero_division=0,
                                            labels=best_cv_final_estimator.classes_ if hasattr(best_cv_final_estimator,
                                                                                               'classes_') else sorted(
                                                np.unique(y_test))))
                print("\n  Confusion Matrix (Test Set - Best CV Model):\n",
                      confusion_matrix(y_test, y_pred_test_best_cv_model,
                                       labels=best_cv_final_estimator.classes_ if hasattr(best_cv_final_estimator,
                                                                                          'classes_') else sorted(
                                           np.unique(y_test))))

                print(f"\n  Saving the best CV model object ({best_cv_model_name}) to {BEST_MODEL_SAVE_PATH}...")
                try:
                    joblib.dump(best_cv_final_estimator, BEST_MODEL_SAVE_PATH)
                    print(f"    Successfully saved best CV model.")
                except Exception as e_save:
                    print(f"    Error saving best CV model object: {e_save}")
            except Exception as e_detail:
                print(
                    f"    ERROR: Failed to evaluate the best CV model ('{best_cv_model_name}') on the test set. Error: {e_detail}")
        else:
            print(
                "  Skipping detailed test set evaluation for best CV model: No best CV model identified or estimator not available.")
else:
    print(
        "\nSteps 4 & 5 (Test Set Evaluations and Detailed Report) were SKIPPED as per PERFORM_FINAL_BEST_MODEL_EVALUATION=False.")

print("\nOrdinal Classification Benchmarking with hyperparameter tuning finished.")
