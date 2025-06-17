# config.py

import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import mord

# --- GENERAL ---
RANDOM_STATE = 42

# --- PATHS ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'clustering_performance_results'
SHAP_RESULTS_DIR = 'shap_results_top_performers'
REPORT_DIR = 'reports'
INPUT_CSV_PATH = 'QuadState_Tornado_DataInputv2.csv'
CLEANED_CSV_PATH = 'cleaned_data_latlong.csv'
PIPELINE_LOG_PATH = 'pipeline.log'
REPORT_FILENAME = 'pipeline_visual_report.pdf'
DETAILED_RESULTS_CSV = os.path.join(BASE_RESULTS_DIR, 'clustering_performance_detailed_results.csv')
BEST_ESTIMATORS_PATH = os.path.join(BASE_RESULTS_DIR, 'best_estimators_per_combo.pkl')
TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.pkl')
PREPROCESSOR_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl')

# --- DATA CLEANING ---
TARGET_COLUMN_FOR_NAN_DROP = 'degree_of_damage_u'
LOW_VARIATION_THRESHOLD = 1
KEYWORDS_TO_DROP = ['photos', 'details', 'prop_', '_unc']
SPECIFIC_COLUMNS_TO_DROP = [
    'completed_by', 'damage_status', 'ref# (DELETE LATER)', 'complete_address',
    'building_name_listing', 'building_name_current', 'notes', 'tornado_name',
    'tornado_EF', 'tornado_start_lat', 'tornado_start_long', 'tornado_end_lat',
    'tornado_end_long', 'national_register_listing_year', 'town',
    'located_in_historic_district', 'hazards_present_u'
]
COLUMNS_FOR_VALUE_REPLACEMENT = {
    'wall_thickness': {'un': '', 'not_applicable': 0},
    'overhang_length_u': {'un': '', 'not_applicable': 0},
    'parapet_height_m': {'un': '', 'not_applicable': 0}
}

# --- PREPROCESSING ---
TARGET_COLUMN = 'degree_of_damage_u'
TEST_SIZE = 0.2
REDUCE_CLASSES_STRATEGY = 'B'
CLASS_MAPPINGS = {
    'A': {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3},
    'B': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
}
BALANCING_METHOD = 'SMOTE'
PERFORM_RFE = True
N_FEATURES_TO_SELECT = 50
KEYWORDS_TO_REMOVE_FROM_X = [
    'damage', 'status_u', 'exist', 'demolish', 'failure', 'after'
]

# --- MODELING & EVALUATION ---
CLUSTERING_THRESHOLDS_TO_TEST = [None]
CLUSTERING_LINKAGE_METHOD = 'average'
N_SPLITS_CV = 5
GRIDSEARCH_SCORING_METRIC = 'f1_weighted'
PERFORMANCE_THRESHOLD_FOR_PLOT = 0.8
N_PERMUTATION_REPEATS = 100  # Added this line
P_VALUE_THRESHOLD = 0.05    # Added this line
METRICS_TO_EVALUATE = {
    'accuracy': 'accuracy', 'f1_weighted': 'f1_weighted', 'f1_macro': 'f1_macro',
    'precision_weighted': 'precision_weighted', 'recall_weighted': 'recall_weighted',
}

MODELS_TO_BENCHMARK = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000, solver='liblinear'),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='mlogloss'),
    "LightGBM": lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
    "Ordinal Logistic (AT)": mord.LogisticAT(),
    "Ordinal Ridge": mord.OrdinalRidge(),
    "Ordinal LAD": mord.LAD()
}

PARAM_GRIDS = {
    "Logistic Regression": {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0]},
    "Decision Tree": {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8], 'min_samples_leaf': [10, 15]},
    "Random Forest": {'n_estimators': [100, 150], 'max_depth': [6, 8], 'min_samples_leaf': [5, 10]},
    "Gradient Boosting": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]},
    "Hist Gradient Boosting": {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [20, 31]},
    "XGBoost": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]},
    "LightGBM": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'num_leaves': [15, 25]},
    "Ordinal Logistic (AT)": {'alpha': [0.1, 1.0, 10.0]},
    "Ordinal Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "Ordinal LAD": {'C': [0.1, 1.0, 10.0]}
}

# --- VISUALIZATION ---
# Define a consistent color scheme for all plots
VISUALIZATION = {
    'main_palette': 'viridis',  # A good choice for sequential data (e.g., bar charts)
    'diverging_palette': 'coolwarm', # Good for heatmaps or SHAP plots where values diverge from a center
    'plot_style': 'seaborn-v0_8-whitegrid' # A clean, professional plot style
}