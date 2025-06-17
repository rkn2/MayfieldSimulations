import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# --- GENERAL ---
RANDOM_STATE = 42

# --- PATHS ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'clustering_performance_results'
SHAP_RESULTS_DIR = 'shap_results_top_performers'
REPORT_DIR = 'reports'
INPUT_CSV_PATH = 'feature_importance_sims.csv' # Using the sims data
CLEANED_CSV_PATH = 'cleaned_data.csv'
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
TARGET_COLUMN_FOR_NAN_DROP = 'Difference'
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
TARGET_COLUMN = 'Difference'
TEST_SIZE = 0.2
REDUCE_CLASSES_STRATEGY = None # Set to None for regression
BALANCING_METHOD = None # Set to None for regression
PERFORM_RFE = True
N_FEATURES_TO_SELECT = 36 # Updated based on your RFE analysis

# This dictionary is a remnant from the classification setup.
# It's included here to prevent an AttributeError in 2_dataPreprocessing.py,
# but it will not be used as long as REDUCE_CLASSES_STRATEGY is None.
CLASS_MAPPINGS = {
    'A': {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3},
    'B': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
}

# This list contains features to remove from the feature set (X)
# because they "leak" information about the outcome or are directly
# related to the simulation outputs we are trying to predict.
KEYWORDS_TO_REMOVE_FROM_X = [
    # Leaky features (know the outcome)
    'demolishing_year',
    'demoshed_by_2023',
    'buidling_use_after_tornado',
    'buidling_use_plan_after_tornado',

    # Simulation/Target-related outputs
    'simulated_damage',
    'estimated',

    # Other general keywords that might relate to damage assessment post-event
    'damage',
    'status_u',
    'exist',
    'demolish',
    'failure',
    'after'
]

# --- MODELING & EVALUATION (Regression) ---
CLUSTERING_THRESHOLDS_TO_TEST = [None]
CLUSTERING_LINKAGE_METHOD = 'average'
N_SPLITS_CV = 5
GRIDSEARCH_SCORING_METRIC = 'neg_mean_squared_error'
PERFORMANCE_THRESHOLD_FOR_PLOT = 0

MODELS_TO_BENCHMARK = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(random_state=RANDOM_STATE),
    "Lasso": Lasso(random_state=RANDOM_STATE),
    "ElasticNet": ElasticNet(random_state=RANDOM_STATE),
    "SVR": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "XGBoost": xgb.XGBRegressor(random_state=RANDOM_STATE),
    "LightGBM": lgb.LGBMRegressor(random_state=RANDOM_STATE, verbosity=-1),
}

PARAM_GRIDS = {
    "Ridge": {'alpha': [0.1, 1.0, 10.0]},
    "Lasso": {'alpha': [0.1, 1.0, 10.0]},
    "ElasticNet": {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.5, 0.9]},
    "SVR": {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10]},
    "Decision Tree": {'max_depth': [4, 6, 8], 'min_samples_leaf': [10, 15]},
    "Random Forest": {'n_estimators': [100, 150], 'max_depth': [6, 8], 'min_samples_leaf': [5, 10]},
    "Gradient Boosting": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]},
    "Hist Gradient Boosting": {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [20, 31]},
    "XGBoost": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4, 5]},
    "LightGBM": {'n_estimators': [100], 'learning_rate': [0.05, 0.1], 'num_leaves': [15, 25]},
}

# --- VISUALIZATION ---
VISUALIZATION = {
    'main_palette': 'viridis',
    'diverging_palette': 'coolwarm',
    'plot_style': 'seaborn-v0_8-white'
}
