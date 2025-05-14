import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
# Updated metrics for classification
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, log_loss, roc_auc_score, make_scorer
)
from sklearn.model_selection import KFold, cross_validate, GridSearchCV

# Import CLASSIFICATION Models
from sklearn.linear_model import LogisticRegression # Replaces LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC # Replaces SVR
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
# Update filename to reflect classification
RESULTS_FILENAME = 'model_tuned_cv_classification_results.csv'

TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
FULL_RESULTS_CSV_PATH = os.path.join(DATA_DIR, RESULTS_FILENAME)
BEST_MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'best_tuned_classifier.pkl') # Updated

RANDOM_STATE = 42

# --- Cross-Validation & Tuning Settings ---
N_SPLITS = 5
# Scoring metric for GridSearchCV to optimize (choose one primary classification metric)
GRIDSEARCH_SCORING = 'f1_weighted' # Good for potentially imbalanced classes
# Other options: 'accuracy', 'f1_macro', 'roc_auc_ovr_weighted', 'neg_log_loss'

# Scoring metrics to calculate *after* finding the best estimator
CV_SCORING_REPORT = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted',
    'recall_weighted': 'recall_weighted',
    # 'neg_log_loss': 'neg_log_loss', # Requires predict_proba
    # 'roc_auc_ovr_weighted': make_scorer(roc_auc_score, needs_proba=True, average='weighted', multi_class='ovr')
}

# Define the base CLASSIFIER models
MODELS_TO_TEST = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, solver='liblinear'), # Good default solver
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    # "SVC (Linear)": SVC(kernel='linear', probability=True, random_state=RANDOM_STATE), # probability=True for ROC AUC if needed, but slower
    # "SVC (RBF)": SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}
if XGB_AVAILABLE:
    MODELS_TO_TEST["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='mlogloss') # Common XGB params
if LGBM_AVAILABLE:
    MODELS_TO_TEST["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)


# --- Parameter Grids for GridSearchCV (adjust for classifiers) ---
PARAM_GRIDS = {
    "Logistic Regression": {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'solver': ['liblinear'] # 'liblinear' supports l1 and l2
        # 'class_weight': [None, 'balanced'] # If classes are imbalanced after reduction
    },
    "Decision Tree": {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    },
    "Random Forest": {
        'n_estimators': [100, 200], # Reduced for speed
        'criterion': ['gini', 'entropy'],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        # 'class_weight': [None, 'balanced', 'balanced_subsample']
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    },
    "Hist Gradient Boosting": {
        'learning_rate': [0.05, 0.1],
        'max_leaf_nodes': [31, 63],
        'max_depth': [None] # Often controlled by max_leaf_nodes
    },
    "SVC (Linear)": {
        'C': [0.1, 1, 10]
    },
    "SVC (RBF)": {
        'C': [1, 10, 100],
        'gamma': ['scale', 0.1, 1]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'euclidean', 'manhattan']
    }
}
if XGB_AVAILABLE:
    PARAM_GRIDS["XGBoost"] = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
if LGBM_AVAILABLE:
    PARAM_GRIDS["LightGBM"] = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [20, 31, 40], # Key parameter for LightGBM
        'max_depth': [-1, 5, 10]
    }

MODELS_TO_TUNE = {name: model for name, model in MODELS_TO_TEST.items() if name in PARAM_GRIDS}

# Define primary metric for comparing models based on CV results
PRIMARY_METRIC = "Mean CV F1 Weighted" # Changed
HIGHER_IS_BETTER = True # F1-score: True

# --- Main Script ---
print("Starting Ordinal Classification Model Benchmarking with Hyperparameter Tuning (GridSearchCV)...") # Updated title
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


print("\nStep 1: Loading preprocessed data...")
try:
    X_train = joblib.load(TRAIN_X_PATH)
    y_train = joblib.load(TRAIN_Y_PATH)
    X_test = joblib.load(TEST_X_PATH)
    y_test = joblib.load(TEST_Y_PATH)
    print(f"  Data loaded successfully from '{DATA_DIR}' directory.")
    print(f"  X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"  X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"  Unique y_train labels: {np.unique(y_train)}")
    print(f"  Unique y_test labels: {np.unique(y_test)}")
except FileNotFoundError as e:
    print(f"Error: Could not find data file: {e}.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Ensure y is 1D array
if isinstance(y_train, (pd.Series, pd.DataFrame)):
    y_train = y_train.values.ravel()
if isinstance(y_test, (pd.Series, pd.DataFrame)):
    y_test = y_test.values.ravel()


print(f"\nStep 2: Running GridSearchCV with {N_SPLITS}-Fold CV for each model...")
tuned_results_list = []
best_estimators = {}
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for model_name, model in MODELS_TO_TUNE.items():
    print(f"  Tuning model: {model_name}...")
    param_grid = PARAM_GRIDS.get(model_name, {}) # Use .get for safety

    # For ROC AUC scoring, we need to handle models that don't have predict_proba
    current_gridsearch_scoring = GRIDSEARCH_SCORING
    current_cv_scoring_report = CV_SCORING_REPORT.copy() # Use a copy to modify

    if 'roc_auc' in GRIDSEARCH_SCORING and not hasattr(model, 'predict_proba'):
        print(f"    Model {model_name} does not support predict_proba. Skipping ROC AUC for GridSearch, using 'f1_weighted'.")
        current_gridsearch_scoring = 'f1_weighted'
    # Remove ROC AUC from report if model doesn't support predict_proba
    if not hasattr(model, 'predict_proba') and 'roc_auc_ovr_weighted' in current_cv_scoring_report:
        del current_cv_scoring_report['roc_auc_ovr_weighted']
        print(f"    Model {model_name} does not support predict_proba. Removing ROC AUC from final CV report.")


    if not param_grid and model_name in MODELS_TO_TEST: # Model has no grid but is in MODELS_TO_TEST
        print("    No parameters to tune for this model in PARAM_GRIDS. Using default and running CV.")
        start_time = time.time()
        try:
            cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=current_cv_scoring_report, n_jobs=-1)
            best_params = {}
            best_estimator = model.fit(X_train, y_train)
            best_estimators[model_name] = best_estimator
        except Exception as e:
             print(f"    ERROR: Model {model_name} failed during default cross-validation. Error: {e}")
             result_row = {"Model": model_name, "Best Params": "N/A", "GridSearch Time (s)": np.nan, "Error": str(e)}
             for metric_name in CV_SCORING_REPORT.keys(): # Use original CV_SCORING_REPORT for column names
                 result_row[f"Mean CV {metric_name.replace('_', ' ').title()}"] = np.nan
             tuned_results_list.append(result_row)
             continue
        tuning_time = time.time() - start_time
    elif model_name in PARAM_GRIDS: # Model has parameters to tune
        start_time = time.time()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=current_gridsearch_scoring,
            cv=kf,
            n_jobs=-1,
            verbose=0
        )
        try:
            grid_search.fit(X_train, y_train)
            tuning_time = time.time() - start_time
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_
            best_estimators[model_name] = best_estimator
            print(f"    Best params found: {best_params}")
            print(f"    Running final CV on best '{model_name}' estimator...")
            cv_results = cross_validate(
                best_estimator, X_train, y_train, cv=kf, scoring=current_cv_scoring_report, n_jobs=-1
            )
        except Exception as e:
            print(f"    ERROR: Model {model_name} failed during GridSearchCV. Error: {e}")
            result_row = {"Model": model_name, "Best Params": "N/A", "GridSearch Time (s)": np.nan, "Error": str(e)}
            for metric_name in CV_SCORING_REPORT.keys():
                 result_row[f"Mean CV {metric_name.replace('_', ' ').title()}"] = np.nan
            tuned_results_list.append(result_row)
            continue
    else:
        print(f"    Skipping {model_name} as it's not in PARAM_GRIDS for tuning and not set for default CV.")
        continue


    # Process results from the cross_validate call
    result_row = {
        "Model": model_name,
        "Best Params": str(best_params),
        "GridSearch Time (s)": tuning_time
    }
    print_metrics = []
    for key, scorer_name in current_cv_scoring_report.items(): # Use potentially modified report
        # Scikit-learn returns 'test_<scorer_key>'
        metric_values = cv_results.get(f'test_{key}', []) # Use .get for safety
        if len(metric_values) > 0:
            mean_metric = np.mean(metric_values)
            # Handle negative scores like neg_log_loss
            if scorer_name.startswith('neg_'):
                mean_metric = -mean_metric
            result_key_name = f"Mean CV {key.replace('_', ' ').title()}"
            result_row[result_key_name] = mean_metric
            print_metrics.append(f"{result_key_name}: {mean_metric:.4f}")
        else:
            result_row[f"Mean CV {key.replace('_', ' ').title()}"] = np.nan # Metric not found

    tuned_results_list.append(result_row)
    print(f"    Finished tuning/CV in {tuning_time:.2f} seconds. Metrics: {', '.join(print_metrics)}")


print("\nStep 3: Summarizing Tuned Cross-Validation results...")
if not tuned_results_list:
    print("No models were successfully tuned or evaluated.")
    exit()

tuned_cv_results_df = pd.DataFrame(tuned_results_list)

if PRIMARY_METRIC not in tuned_cv_results_df.columns:
     print(f"Error: Primary metric '{PRIMARY_METRIC}' not found in results columns: {tuned_cv_results_df.columns.tolist()}")
     print("Available columns:", tuned_cv_results_df.columns)
     # Fallback to a common metric if primary is missing, e.g., F1 Weighted
     if "Mean CV F1 Weighted" in tuned_cv_results_df.columns:
         PRIMARY_METRIC = "Mean CV F1 Weighted"
         print(f"Falling back to '{PRIMARY_METRIC}' as primary metric.")
     elif "Mean CV Accuracy" in tuned_cv_results_df.columns:
         PRIMARY_METRIC = "Mean CV Accuracy"
         print(f"Falling back to '{PRIMARY_METRIC}' as primary metric.")
     else:
         print("Could not find a suitable fallback primary metric. Exiting.")
         exit()


tuned_cv_results_df_sorted = tuned_cv_results_df.sort_values(
    by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER, na_position='last'
).copy()

print(f"\nStep 3.5: Saving tuned CV results summary to {FULL_RESULTS_CSV_PATH}...")
try:
    target_dir = os.path.dirname(FULL_RESULTS_CSV_PATH)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    tuned_cv_results_df_sorted.to_csv(FULL_RESULTS_CSV_PATH, index=False, float_format='%.6f')
    print(f"  Successfully saved tuned CV results to {FULL_RESULTS_CSV_PATH}")
except Exception as e:
    print(f"Error: Could not save tuned CV results to CSV at {FULL_RESULTS_CSV_PATH}. Error: {e}")

print("\n--- Tuned Cross-Validation Model Comparison (Display) ---")
tuned_cv_results_df_display = tuned_cv_results_df_sorted.copy()
numeric_cols_display = tuned_cv_results_df_display.select_dtypes(include=np.number).columns
for col in numeric_cols_display:
     if col in tuned_cv_results_df_display:
          tuned_cv_results_df_display[col] = tuned_cv_results_df_display[col].apply(
              lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
          )
print(tuned_cv_results_df_display.to_string(index=False))


print("\nStep 4: Identifying best tuned model based on Cross-Validation...")
valid_tuned_results = pd.DataFrame(tuned_results_list).dropna(subset=[PRIMARY_METRIC])
best_model_name = None
best_final_estimator = None

if not valid_tuned_results.empty:
     valid_tuned_results_sorted = valid_tuned_results.sort_values(
         by=PRIMARY_METRIC, ascending=not HIGHER_IS_BETTER
     )
     best_model_row = valid_tuned_results_sorted.iloc[0]
     best_model_name = best_model_row['Model']
     if best_model_name in best_estimators:
         best_final_estimator = best_estimators[best_model_name]
         print(f"\nBest model type based on {PRIMARY_METRIC} ({'higher' if HIGHER_IS_BETTER else 'lower'} is better):")
         print(f"  Model: {best_model_row['Model']}")
         print(f"  Best Params: {best_model_row['Best Params']}")
         # Ensure PRIMARY_METRIC value is numeric for formatting if it came from DataFrame
         primary_metric_value = best_model_row[PRIMARY_METRIC]
         if isinstance(primary_metric_value, str) and primary_metric_value != "N/A":
             try: primary_metric_value = float(primary_metric_value)
             except ValueError: pass # Keep as string if not convertible
         print(f"  {PRIMARY_METRIC}: {primary_metric_value:.4f}" if isinstance(primary_metric_value, float) else f"  {PRIMARY_METRIC}: {primary_metric_value}")
         print(f"  Tuning/CV Time: {float(best_model_row['GridSearch Time (s)']):.2f} seconds")

     else:
          print(f"Error: Best model '{best_model_name}' not found in stored estimators dictionary.")
          best_model_name = None
else:
     print("Could not determine the best model from tuning results (no valid results).")


print("\nStep 5: Final evaluation of the best tuned model on the hold-out test set...")
if best_model_name and best_final_estimator:
    print(f"  Using best estimator for '{best_model_name}' (already fitted on full training data)...")
    try:
        y_pred_test = best_final_estimator.predict(X_test)
        # For classification, get probabilities if model supports it and you need them (e.g., for log_loss, ROC AUC on test)
        y_pred_proba_test = None
        if hasattr(best_final_estimator, "predict_proba"):
            y_pred_proba_test = best_final_estimator.predict_proba(X_test)

        print("\n  --- Final Test Set Performance (Best Tuned Model) ---")
        print(f"  Model: {best_model_name}")
        print(f"  Params Used: {best_final_estimator.get_params() if hasattr(best_final_estimator, 'get_params') else 'N/A'}")

        # Classification Metrics
        print(f"  Test Accuracy:        {accuracy_score(y_test, y_pred_test):.4f}")
        print(f"  Test F1 Score (Weighted): {f1_score(y_test, y_pred_test, average='weighted'):.4f}")
        print(f"  Test F1 Score (Macro):    {f1_score(y_test, y_pred_test, average='macro'):.4f}")
        if y_pred_proba_test is not None and len(np.unique(y_test)) > 1 : # Check for multi-class and if probabilities exist
            try:
                # Ensure y_test has all classes present in y_pred_proba_test if using OvR/OvO for roc_auc
                # Or ensure labels are consistent if some classes from training are missing in test
                labels_for_roc_auc = sorted(np.unique(y_test)) # Or np.unique(np.concatenate((y_train, y_test)))
                if len(labels_for_roc_auc) > 1 and y_pred_proba_test.shape[1] >= len(labels_for_roc_auc):
                     # Adjust predict_proba if its columns don't match unique labels in y_test for some reason
                     # This can happen if a class is in y_train but not y_test for roc_auc_score with multi_class='ovr'
                     # For simplicity, this example assumes y_pred_proba_test columns align with sorted unique classes in y_train
                     # A more robust way is to pass `labels=best_final_estimator.classes_` to roc_auc_score
                     if y_pred_proba_test.shape[1] == len(best_final_estimator.classes_):
                         print(f"  Test ROC AUC (OvR Wtd): {roc_auc_score(y_test, y_pred_proba_test, average='weighted', multi_class='ovr', labels=best_final_estimator.classes_):.4f}")
                         print(f"  Test Log Loss:          {log_loss(y_test, y_pred_proba_test, labels=best_final_estimator.classes_):.4f}")
                     else:
                         print("  Skipping ROC AUC/Log Loss on test set: mismatch in number of classes for probabilities.")

            except Exception as e_roc:
                print(f"    Could not calculate ROC AUC or Log Loss on test set: {e_roc}")


        print("\n  Classification Report (Test Set):\n", classification_report(y_test, y_pred_test, zero_division=0))
        print("\n  Confusion Matrix (Test Set):\n", confusion_matrix(y_test, y_pred_test))

        print(f"\n  Saving the best fitted model object to {BEST_MODEL_SAVE_PATH}...")
        try:
            joblib.dump(best_final_estimator, BEST_MODEL_SAVE_PATH)
            print(f"    Successfully saved best model.")
        except Exception as e:
            print(f"    Error saving best model object: {e}")
    except Exception as e:
        print(f"    ERROR: Failed to evaluate the best tuned model ('{best_model_name}') on the test set. Error: {e}")
else:
    print("  Skipping final evaluation: No best tuned model identified or best estimator object not available.")

print("\nOrdinal Classification Benchmarking with hyperparameter tuning finished.")