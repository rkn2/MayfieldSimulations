import pandas as pd
import numpy as np
import os
import joblib
import warnings
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import config  # Import the configuration file

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Set the random seed for reproducibility
np.random.seed(config.RANDOM_STATE)


# --- Logging Configuration Setup ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    """Sets up logging to append to the main pipeline log file."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # 'a' for append
            logging.StreamHandler(sys.stdout)
        ]
    )


# Configure logging
setup_logging()

# --- Dynamic Importer for Balancing Methods ---
sampler_class = None
if config.BALANCING_METHOD:
    try:
        if config.BALANCING_METHOD == 'RandomOverSampler':
            from imblearn.over_sampling import RandomOverSampler

            sampler_class = RandomOverSampler
        elif config.BALANCING_METHOD == 'SMOTE':
            from imblearn.over_sampling import SMOTE

            sampler_class = SMOTE
        logging.info(f"Selected balancing method: {config.BALANCING_METHOD}")
    except ImportError:
        logging.error(f"Error: 'imbalanced-learn' not found for BALANCING_METHOD '{config.BALANCING_METHOD}'.")
        logging.error("Please install it: pip install imbalanced-learn")
        sys.exit()


def filter_features(df, keywords_to_remove):
    """Removes columns from the dataframe based on a list of keywords."""
    cols_to_drop = {col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords_to_remove)}
    logging.info(f"Filtering features. Removing columns based on keywords: {sorted(list(cols_to_drop))}")
    return df.drop(columns=list(cols_to_drop), errors='ignore')


# --- Main Preprocessing Script ---

logging.info(f"--- Starting Script: 2_dataPreprocessing.py ---")
logging.info(f"Class Reduction Strategy: {config.REDUCE_CLASSES_STRATEGY or 'None'}")
logging.info(f"Balancing Method: {config.BALANCING_METHOD or 'None'}")

# 1. Load Data
logging.info(f"\nStep 1: Loading cleaned data from '{config.CLEANED_CSV_PATH}'...")
try:
    df = pd.read_csv(config.CLEANED_CSV_PATH, low_memory=False)
    logging.info(f"  Successfully loaded. Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Cleaned data file not found at {config.CLEANED_CSV_PATH}")
    sys.exit()

# 2. Separate Target and Features
logging.info(f"\nStep 2: Separating target ('{config.TARGET_COLUMN}') and features...")
if config.TARGET_COLUMN not in df.columns:
    logging.error(f"Error: Target column '{config.TARGET_COLUMN}' not found.")
    sys.exit()
y = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce').fillna(0).astype(int)
X = df.drop(columns=[config.TARGET_COLUMN])

# 3. Apply Class Reduction
logging.info("\nStep 3: Applying class reduction...")
if config.REDUCE_CLASSES_STRATEGY in config.CLASS_MAPPINGS:
    y = y.map(config.CLASS_MAPPINGS[config.REDUCE_CLASSES_STRATEGY])
    logging.info(f"  Applied class reduction strategy '{config.REDUCE_CLASSES_STRATEGY}'.")
    logging.info(f"  New y distribution:\n{y.value_counts(normalize=True).sort_index()}")
else:
    logging.info("  No class reduction strategy applied.")

# 4. Filter Feature Set
logging.info("\nStep 4: Filtering feature set (X) based on keywords...")
X = filter_features(X, config.KEYWORDS_TO_REMOVE_FROM_X)
logging.info(f"  Shape of X after keyword filtering: {X.shape}")

# 5. Split Data
logging.info("\nStep 5: Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                                                    stratify=y)
logging.info(f"  Data split successfully (stratified).")
logging.info(f"  y_train distribution (before balancing):\n{y_train.value_counts(normalize=True).sort_index()}")

# 6. Preprocess Features
logging.info("\nStep 6: Preprocessing features (scaling and one-hot encoding)...")
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features = X_train.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Convert to DataFrame
feature_names_out = preprocessor.get_feature_names_out()
X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
logging.info(f"  Preprocessing complete. Shape of processed X_train: {X_train_processed_df.shape}")

# 7. Recursive Feature Elimination (RFE)
if config.PERFORM_RFE:
    logging.info(f"\nStep 7: Performing RFE to select top {config.N_FEATURES_TO_SELECT} features...")
    rfe_estimator = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    selector = RFE(estimator=rfe_estimator, n_features_to_select=config.N_FEATURES_TO_SELECT, step=0.1)
    selector.fit(X_train_processed_df, y_train)

    selected_feature_names = X_train_processed_df.columns[selector.support_]
    X_train_processed_df = X_train_processed_df[selected_feature_names]
    X_test_processed_df = X_test_processed_df[selected_feature_names]
    logging.info(f"  RFE complete. New shape of X_train: {X_train_processed_df.shape}")

# 8. Balance Training Data
X_train_final = X_train_processed_df
y_train_final = y_train

if config.BALANCING_METHOD and sampler_class:
    logging.info(f"\nStep 8: Applying balancing method '{config.BALANCING_METHOD}'...")
    min_class_size = y_train.value_counts().min()
    k_neighbors = max(1, min_class_size - 1) if config.BALANCING_METHOD == 'SMOTE' and min_class_size <= 5 else 5
    sampler = sampler_class(random_state=config.RANDOM_STATE,
                            k_neighbors=k_neighbors) if config.BALANCING_METHOD == 'SMOTE' else sampler_class(
        random_state=config.RANDOM_STATE)

    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed_df, y_train)
    X_train_final = pd.DataFrame(X_train_resampled, columns=X_train_processed_df.columns)
    y_train_final = y_train_resampled
    logging.info(f"  Resampling complete. New shape of X_train: {X_train_final.shape}")
    logging.info(f"  y_train distribution (after balancing):\n{y_train_final.value_counts().sort_index()}")

# 9. Save Processed Data
logging.info("\nStep 9: Saving final processed data...")
os.makedirs(config.DATA_DIR, exist_ok=True)
joblib.dump(X_train_final, config.TRAIN_X_PATH)
joblib.dump(y_train_final, config.TRAIN_Y_PATH)
joblib.dump(X_test_processed_df, config.TEST_X_PATH)
joblib.dump(y_test, config.Y_TEST_PATH)
joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
logging.info(f"  Data saved to '{config.DATA_DIR}' directory.")

logging.info(f"\n--- Finished Script: 2_dataPreprocessing.py ---")