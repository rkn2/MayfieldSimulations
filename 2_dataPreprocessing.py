import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import os
import joblib
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# --- Logging Configuration Setup ---
def setup_logging(log_file='pipeline.log'):
    """Sets up logging to both a file and the console."""
    # This will append to the file created by the first script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'), # 'a' for append
            logging.StreamHandler(sys.stdout)
        ]
    )

# Call the setup function
setup_logging()


# --- Configuration ---
SUBSAMPLE_DAMAGE_0 = 50  # Number of rows to keep for damage level 0.  Set to None to disable.

INPUT_CSV_PATH = 'cleaned_data_latlong.csv'  # Output from the previous cleaning script
TARGET_COLUMN = 'degree_of_damage_u'

# --- Class Reduction Configuration ---
REDUCE_CLASSES_STRATEGY = 'B'  # Options: 'A', 'B', None, etc.
CLASS_MAPPINGS = {
    'A': {0: 0, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3},
    'B': {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2}
}
NEW_CATEGORY_ORDER = {
    'A': [0, 1, 2, 3],
    'B': [0, 1, 2]
}
NEW_CATEGORY_LABELS = {
    'A': ['Undamaged (0)', 'Low-Mod (1)', 'Significant (2)', 'Demolished (3)'],
    'B': ['Undamaged (0)', 'Repairable (1)', 'Demolished (2)']
}

# --- Balancing Configuration ---
BALANCING_METHOD = 'SMOTE'
SAMPLING_STRATEGY = 'auto'

KEYWORDS_TO_REMOVE_FROM_X = [
    'damage', 'status_u', 'exist', 'demolish', 'failure', 'after'
]
TEST_SIZE = 0.2
RANDOM_STATE = 42

SAVE_PROCESSED_DATA = True
OUTPUT_DIR = 'processed_ml_data'
OUTPUT_TRAIN_X_PATH = f'{OUTPUT_DIR}/X_train_processed.pkl'
OUTPUT_TEST_X_PATH = f'{OUTPUT_DIR}/X_test_processed.pkl'
OUTPUT_TRAIN_Y_PATH = f'{OUTPUT_DIR}/y_train.pkl'
OUTPUT_TEST_Y_PATH = f'{OUTPUT_DIR}/y_test.pkl'
OUTPUT_PREPROCESSOR_PATH = f'{OUTPUT_DIR}/preprocessor.pkl'

sampler_class = None
if BALANCING_METHOD in ['RandomOverSampler', 'SMOTE']:
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE

        if BALANCING_METHOD == 'RandomOverSampler':
            sampler_class = RandomOverSampler
            logging.info("Selected balancing method: RandomOverSampler")
        elif BALANCING_METHOD == 'SMOTE':
            sampler_class = SMOTE
            logging.info("Selected balancing method: SMOTE")
    except ImportError:
        logging.error(f"Error: 'imbalanced-learn' library not found, but BALANCING_METHOD is set to '{BALANCING_METHOD}'.")
        logging.error("Please install it: pip install imbalanced-learn")
        exit()
elif BALANCING_METHOD is not None:
    logging.warning(f"Unknown BALANCING_METHOD '{BALANCING_METHOD}'. Balancing will be disabled.")
    BALANCING_METHOD = None


def filter_features(df, target_col, keywords_to_remove):
    cols_to_drop = set()
    keywords_lower = [kw.lower() for kw in keywords_to_remove]
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords_lower):
            cols_to_drop.add(col)
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    logging.info("Filtering features. Removing columns based on keywords (target column already separated):")
    logging.info(f"  Keywords: {keywords_to_remove}")
    logging.info(f"  Columns identified for removal from X: {sorted(list(cols_to_drop_existing))}")
    X = df.drop(columns=cols_to_drop_existing, errors='ignore')
    logging.info(f"  Number of features remaining in X: {X.shape[1]}")
    return X


logging.info(f"--- Starting Script: 2_dataPreprocessing.py ---")
logging.info(f"Starting ML preprocessing for {INPUT_CSV_PATH}")
logging.info(f"Selected Class Reduction Strategy: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'}")
logging.info(f"Selected Balancing Method: {BALANCING_METHOD if BALANCING_METHOD else 'None'}")
if BALANCING_METHOD:
    logging.info(f"Sampling Strategy: {SAMPLING_STRATEGY}")

logging.info("\nStep 1: Loading cleaned data...")
try:
    df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
    logging.info(f"  Successfully loaded. Shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Cleaned data file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    logging.error(f"Error loading CSV: {e}")
    exit()

if SUBSAMPLE_DAMAGE_0 is not None and (REDUCE_CLASSES_STRATEGY is None or (
        REDUCE_CLASSES_STRATEGY and CLASS_MAPPINGS.get(REDUCE_CLASSES_STRATEGY, {}).get(0) == 0)):
    logging.info(f"\nStep 2: Subsampling original damage level 0 to {SUBSAMPLE_DAMAGE_0} rows...")
    damage_0_indices = df[df[TARGET_COLUMN] == 0].index
    if len(damage_0_indices) > SUBSAMPLE_DAMAGE_0:
        random_indices = np.random.choice(damage_0_indices, SUBSAMPLE_DAMAGE_0, replace=False)
        other_indices = df.index.difference(damage_0_indices)
        df = df.loc[pd.Index(random_indices).union(other_indices)]
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        logging.info(f"  Subsampled DataFrame shape: {df.shape}")
    else:
        logging.info(
            f"  Original damage level 0 has {len(damage_0_indices)} rows, which is <= {SUBSAMPLE_DAMAGE_0}. Skipping subsampling.")
elif SUBSAMPLE_DAMAGE_0 is not None:
    logging.info(
        f"\nStep 2: Subsampling of original damage level 0 skipped because it's being remapped by strategy {REDUCE_CLASSES_STRATEGY}.")

logging.info(f"\nStep 3: Separating target variable '{TARGET_COLUMN}'...")
if TARGET_COLUMN not in df.columns:
    logging.error(f"Error: Target column '{TARGET_COLUMN}' not found in the loaded data.")
    exit()
y_original = df[TARGET_COLUMN]
try:
    y = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).astype(int)
    logging.info(f"  Target variable '{TARGET_COLUMN}' separated and converted to integer type.")
    logging.info(f"  Original y distribution (after any subsampling):\n{y.value_counts(normalize=True).sort_index()}")
except Exception as e:
    logging.warning(f"Could not convert target variable '{TARGET_COLUMN}' to integer. Error: {e}")
    y_original = y.copy()

logging.info("\nStep 3.5: Applying Class Reduction...")
if REDUCE_CLASSES_STRATEGY and REDUCE_CLASSES_STRATEGY in CLASS_MAPPINGS:
    mapping_to_apply = CLASS_MAPPINGS[REDUCE_CLASSES_STRATEGY]
    y_reduced = y.map(mapping_to_apply)
    logging.info(f"  Applied class reduction strategy '{REDUCE_CLASSES_STRATEGY}'.")
    logging.info(f"  y distribution BEFORE reduction (from current df):\n{y.value_counts(normalize=True).sort_index()}")
    logging.info(f"  y distribution AFTER reduction:\n{y_reduced.value_counts(normalize=True).sort_index()}")
    y = y_reduced
    current_category_order = NEW_CATEGORY_ORDER.get(REDUCE_CLASSES_STRATEGY, sorted(y.unique()))
    current_category_labels = NEW_CATEGORY_LABELS.get(REDUCE_CLASSES_STRATEGY, [str(i) for i in current_category_order])
else:
    logging.info("  No class reduction strategy applied or strategy not found.")
    current_category_order = sorted(y.unique())
    current_category_labels = [str(i) for i in current_category_order]

logging.info("\nStep 4: Filtering feature set (X)...")
X = filter_features(df.drop(columns=[TARGET_COLUMN]), TARGET_COLUMN, KEYWORDS_TO_REMOVE_FROM_X)
logging.info(f"  Shape of initial feature set X: {X.shape}")

logging.info("\nStep 5: Identifying numeric and categorical features in the filtered X...")
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
all_features = numeric_features + categorical_features
if len(all_features) != X.shape[1]:
    warnings.warn("Mismatch between identified features and total columns in X.")
logging.info(f"  Identified {len(numeric_features)} numeric features.")
logging.info(f"  Identified {len(categorical_features)} categorical features.")

logging.info("\nStep 6: Defining preprocessing steps...")
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
logging.info("  Preprocessor defined (StandardScaler for numeric, OneHotEncoder for categorical).")

logging.info("\nStep 7: Splitting data into training and testing sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logging.info(f"  Data split successfully (stratified on {'reduced' if REDUCE_CLASSES_STRATEGY else 'original'} classes):")
except ValueError as e:
    logging.warning(f"Could not stratify split (error: {e}). Splitting without stratification.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logging.info(f"  Data split successfully (without stratification):")

logging.info(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logging.info(f"    X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
logging.info(
    f"    y_train distribution (after reduction, before balancing):\n{y_train.value_counts(normalize=True).sort_index()}")
logging.info(f"    y_test distribution (after reduction):\n{y_test.value_counts(normalize=True).sort_index()}")

logging.info("\nStep 8: Applying preprocessing (fitting on train, transforming train and test)...")
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names_out = None
try:
    feature_names_out = preprocessor.get_feature_names_out()
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
    logging.info("  Preprocessing applied. Converted processed arrays to DataFrames.")
except Exception as e:
    logging.warning(f"Could not get feature names or create DataFrames from processed arrays. Error: {e}")
    logging.info("  Proceeding with NumPy arrays for processed X data.")
    X_train_processed_df = X_train_processed
    X_test_processed_df = X_test_processed

logging.info(f"  Shape after preprocessing:")
logging.info(f"    X_train_processed shape: {X_train_processed_df.shape}")
logging.info(f"    X_test_processed shape:  {X_test_processed_df.shape}")

X_train_final = X_train_processed_df
y_train_final = y_train

if BALANCING_METHOD and sampler_class:
    logging.info(f"  Balancing is ENABLED using {BALANCING_METHOD} (strategy='{SAMPLING_STRATEGY}')...")
    sampler = None
    try:
        if BALANCING_METHOD == 'SMOTE':
            min_class_size = y_train.value_counts().min()
            if min_class_size <= 5:
                k_neighbors_val = max(1, min_class_size - 1)
                logging.info(f"  Smallest class has {min_class_size} samples. Setting SMOTE k_neighbors to {k_neighbors_val}.")
                sampler = sampler_class(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE,
                                        k_neighbors=k_neighbors_val)
            else:
                sampler = sampler_class(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
        else:
            sampler = sampler_class(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)

        logging.info(f"  Instantiated {sampler.__class__.__name__} sampler.")

        logging.info(f"  Applying {BALANCING_METHOD}.fit_resample to training data...")
        X_train_to_resample = X_train_processed_df.to_numpy() if hasattr(X_train_processed_df,
                                                                         'to_numpy') else X_train_processed_df
        X_train_resampled_np, y_train_resampled_np = sampler.fit_resample(X_train_to_resample, y_train)

        logging.info(f"  Resampling complete.")
        logging.info(f"    Original y_train distribution (after reduction):\n{y_train.value_counts().sort_index()}")
        y_train_resampled_series = pd.Series(y_train_resampled_np, name=y_train.name)
        logging.info(f"    Resampled y_train distribution:\n{y_train_resampled_series.value_counts().sort_index()}")
        logging.info(f"    X_train_resampled shape: {X_train_resampled_np.shape}")

        y_train_final = y_train_resampled_series
        if isinstance(X_train_processed_df, pd.DataFrame) and feature_names_out is not None:
            try:
                X_train_final = pd.DataFrame(X_train_resampled_np, columns=feature_names_out)
                logging.info("  Converted resampled X_train back to DataFrame.")
            except Exception as e:
                logging.warning(f"Could not convert resampled X_train back to DataFrame: {e}. Keeping as NumPy array.")
                X_train_final = X_train_resampled_np
        else:
            X_train_final = X_train_resampled_np

    except Exception as e:
        logging.error(f"Error during {BALANCING_METHOD} fit_resample: {e}")
        logging.info("  Balancing failed. Using original processed training data (after reduction).")
        X_train_final = X_train_processed_df
        y_train_final = y_train
        BALANCING_METHOD = None
else:
    if BALANCING_METHOD:
        logging.info(f"  Balancing method '{BALANCING_METHOD}' was specified but could not be applied.")
    logging.info("  Using training data (after reduction, no balancing applied).")


logging.info("\nStep 9: Saving final data...")
if SAVE_PROCESSED_DATA:
    logging.info(f"  Saving processed data and preprocessor to '{OUTPUT_DIR}'...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        joblib.dump(X_train_final, OUTPUT_TRAIN_X_PATH)
        joblib.dump(y_train_final, OUTPUT_TRAIN_Y_PATH)
        joblib.dump(X_test_processed_df, OUTPUT_TEST_X_PATH)
        joblib.dump(y_test, OUTPUT_TEST_Y_PATH)
        joblib.dump(preprocessor, OUTPUT_PREPROCESSOR_PATH)
        logging.info(f"  Successfully saved:")
        applied_balancing_status = BALANCING_METHOD if 'y_train_resampled_series' in locals() and y_train_final.shape == y_train_resampled_series.shape else 'None'
        logging.info(
            f"    - Train X: {OUTPUT_TRAIN_X_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'}, Balancing: {applied_balancing_status})")
        logging.info(
            f"    - Train y: {OUTPUT_TRAIN_Y_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'}, Balancing: {applied_balancing_status})")
        logging.info(
            f"    - Test X:  {OUTPUT_TEST_X_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'})")
        logging.info(
            f"    - Test y:  {OUTPUT_TEST_Y_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'})")
        logging.info(f"    - Preprocessor: {OUTPUT_PREPROCESSOR_PATH}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
else:
    logging.info("  Skipping saving processed data (SAVE_PROCESSED_DATA is False).")

logging.info("\nML preprocessing finished.")

# --- Logging and Visualization Code (Subplots) ---
logging.info("\n--- Logging Visualization Data ---")

# Log the data for each plot
original_plot_y = pd.read_csv(INPUT_CSV_PATH, low_memory=False)[TARGET_COLUMN].astype(int)
logging.info(f"Data for 'True Original Data' plot:\n{original_plot_y.value_counts().sort_index().to_string()}")
logging.info(f"Data for 'Training Data (Before Balancing)' plot:\n{y_train.value_counts().sort_index().to_string()}")
applied_balancing_status_log = BALANCING_METHOD if 'y_train_resampled_series' in locals() and y_train_final.shape == y_train_resampled_series.shape else 'None'
logging.info(f"Data for 'Training Data (After Balancing: {applied_balancing_status_log})' plot:\n{y_train_final.value_counts().sort_index().to_string()}")
logging.info(f"Data for 'Test Data' plot:\n{y_test.value_counts().sort_index().to_string()}")

# Generate the plot
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(
    f'Distribution of Degree of Damage (Strategy: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else "Original"})',
    fontsize=16)

sns.countplot(x=original_plot_y, palette='viridis', ax=axes[0, 0], order=list(range(6)))
axes[0, 0].set_title('True Original Data (Before Any Processing)')
axes[0, 0].set_xlabel('Degree of Damage')
axes[0, 0].set_ylabel('Number of Occurrences')
axes[0, 0].set_xticks(ticks=range(6), labels=[str(i) for i in range(6)])
axes[0, 0].set_xlim(-0.5, 5.5)

sns.countplot(x=y_train, palette='viridis', ax=axes[0, 1], order=current_category_order)
axes[0, 1].set_title('Training Data (After Reduction, Before Balancing)')
axes[0, 1].set_xlabel('Degree of Damage')
axes[0, 1].set_ylabel('Number of Occurrences')
axes[0, 1].set_xticks(ticks=current_category_order, labels=current_category_labels)
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].set_xlim(min(current_category_order) - 0.5, max(current_category_order) + 0.5)

sns.countplot(x=y_train_final, palette='viridis', ax=axes[1, 0], order=current_category_order)
axes[1, 0].set_title(f'Training Data (After Reduction & Balancing: {applied_balancing_status_log})')
axes[1, 0].set_xlabel('Degree of Damage')
axes[1, 0].set_ylabel('Number of Occurrences')
axes[1, 0].set_xticks(ticks=current_category_order, labels=current_category_labels)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_xlim(min(current_category_order) - 0.5, max(current_category_order) + 0.5)

sns.countplot(x=y_test, palette='viridis', ax=axes[1, 1], order=current_category_order)
axes[1, 1].set_title('Test Data (After Reduction)')
axes[1, 1].set_xlabel('Degree of Damage')
axes[1, 1].set_ylabel('Number of Occurrences')
axes[1, 1].set_xticks(ticks=current_category_order, labels=current_category_labels)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].set_xlim(min(current_category_order) - 0.5, max(current_category_order) + 0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_plot_filename = f'degree_of_damage_subplots_strategy_{REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else "orig"}.png'
plt.savefig(output_plot_filename)
logging.info(f"Saved plot to {output_plot_filename}")
plt.show()

logging.info(f"--- Finished Script: 2_dataPreprocessing.py ---")