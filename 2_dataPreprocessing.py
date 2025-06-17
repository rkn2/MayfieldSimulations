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
        ],
        force=True
    )


# Configure logging
setup_logging()

# --- Dynamic Importer for Balancing Methods ---
sampler_class = None
if config.BALANCING_METHOD:
    try:
        if config.BALANCING_METHOD == 'SMOTE':
            from imblearn.over_sampling import SMOTE

            sampler_class = SMOTE
        # Add other balancers like RandomOverSampler here if needed
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
def main():
    logging.info(f"--- Starting Script: 2_dataPreprocessing.py ---")

    # 1. Load Data
    logging.info(f"\nStep 1: Loading cleaned data from '{config.CLEANED_CSV_PATH}'...")
    try:
        df = pd.read_csv(config.CLEANED_CSV_PATH, low_memory=False)
        logging.info(f"  Successfully loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"FATAL: Cleaned data file not found at {config.CLEANED_CSV_PATH}")
        sys.exit(1)

    # 2. Separate Target and Features
    logging.info(f"\nStep 2: Separating target ('{config.TARGET_COLUMN}') and features...")
    y = pd.to_numeric(df[config.TARGET_COLUMN], errors='coerce').fillna(0).astype(int)
    X = df.drop(columns=[config.TARGET_COLUMN])

    # 3. Apply Class Reduction
    logging.info("\nStep 3: Applying class reduction...")
    if config.REDUCE_CLASSES_STRATEGY in config.CLASS_MAPPINGS:
        y = y.map(config.CLASS_MAPPINGS[config.REDUCE_CLASSES_STRATEGY])
        logging.info(f"  Applied class reduction strategy '{config.REDUCE_CLASSES_STRATEGY}'.")

    # 4. Filter Feature Set
    X = filter_features(X, config.KEYWORDS_TO_REMOVE_FROM_X)

    # 5. Split Data
    logging.info("\nStep 5: Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE,
                                                        random_state=config.RANDOM_STATE, stratify=y)

    # 6. Preprocess Features
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
        remainder='passthrough')

    X_train_processed = pd.DataFrame(preprocessor.fit_transform(X_train), columns=preprocessor.get_feature_names_out(),
                                     index=X_train.index)
    X_test_processed = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out(),
                                    index=X_test.index)

    # 7. Recursive Feature Elimination (RFE)
    if config.PERFORM_RFE:
        logging.info(f"\nStep 7: Performing RFE to select top {config.N_FEATURES_TO_SELECT} features...")
        selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1),
                       n_features_to_select=config.N_FEATURES_TO_SELECT, step=0.1)
        selector.fit(X_train_processed, y_train)
        X_train_processed = X_train_processed.loc[:, selector.support_]
        X_test_processed = X_test_processed.loc[:, selector.support_]

    # 8. Balance Training Data
    X_train_final, y_train_final = X_train_processed, y_train
    if config.BALANCING_METHOD and sampler_class:
        logging.info(f"\nStep 8: Applying balancing method '{config.BALANCING_METHOD}'...")
        sampler = sampler_class(random_state=config.RANDOM_STATE)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed, y_train)
        X_train_final = pd.DataFrame(X_train_resampled, columns=X_train_processed.columns)
        y_train_final = y_train_resampled

    # 9. Save Data
    logging.info("\nStep 9: Saving final processed data...")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    joblib.dump(X_train_final, config.TRAIN_X_PATH)
    joblib.dump(y_train_final, config.TRAIN_Y_PATH)
    joblib.dump(X_test_processed, config.TEST_X_PATH)
    joblib.dump(y_test, config.Y_TEST_PATH)
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)

    # 10. Visualize Distributions
    logging.info("\nStep 10: Visualizing data distributions...")
    plt.style.use(config.VISUALIZATION['plot_style'])
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    palette = config.VISUALIZATION['main_palette']

    sns.countplot(x=df[config.TARGET_COLUMN], ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title('Original Data Distribution')

    sns.countplot(x=y_train, ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title('Training Data (Before Balancing)')

    sns.countplot(x=y_train_final, ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title(f'Training Data (After {config.BALANCING_METHOD or "No"} Balancing)')

    sns.countplot(x=y_test, ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title('Test Data Distribution')

    plt.tight_layout()
    plt.savefig('data_distribution_summary.png')
    logging.info("Saved data distribution summary plot.")

    logging.info(f"--- Finished Script: 2_dataPreprocessing.py ---")


if __name__ == '__main__':
    main()