import pandas as pd
import numpy as np
import logging
import sys
import os
import config  # Import the configuration file

# Set the random seed for reproducibility
np.random.seed(0)


# --- Logging Configuration Setup ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    """Sets up logging to both a file and the console, appending to the log file."""
    logger = logging.getLogger()

    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler (mode 'a' for append)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


# Configure logging
setup_logging()


# --- Helper Functions ---

def drop_columns_by_keywords(df, keywords):
    """Drops columns containing any of the specified keywords (case-insensitive)."""
    cols_to_drop = [col for col in df.columns if any(keyword.lower() in col.lower() for keyword in keywords)]
    if cols_to_drop:
        logging.info(f"Dropping {len(cols_to_drop)} columns based on keywords: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        logging.info("No columns found matching keyword criteria.")
    return df


def drop_specific_columns(df, columns_list):
    """Drops columns specified in the list if they exist in the DataFrame."""
    cols_to_drop = [col for col in columns_list if col in df.columns]
    if cols_to_drop:
        logging.info(f"Dropping {len(cols_to_drop)} specific columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        logging.info("No specific columns from the list were found to drop.")
    return df


def drop_older_version_columns(df):
    """Drops '_o' columns if a corresponding '_u' column exists."""
    cols_to_drop = [u_col[:-2] + '_o' for u_col in df.columns if
                    u_col.endswith('_u') and u_col[:-2] + '_o' in df.columns]
    if cols_to_drop:
        logging.info(f"Dropping {len(cols_to_drop)} older ('_o') version columns.")
        df.drop(columns=cols_to_drop, inplace=True)
    else:
        logging.info("No older version ('_o') columns found to drop.")
    return df


# --- Main Data Cleaning Script ---

def main():
    logging.info(f"--- Starting Script: 1_dataCleaning.py ---")

    # 1. Load Data
    logging.info(f"Step 1: Loading data from {config.INPUT_CSV_PATH}...")
    try:
        df = pd.read_csv(config.INPUT_CSV_PATH, delimiter=',', encoding='latin-1', low_memory=False)
        logging.info(f"  Successfully loaded. Initial shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at {config.INPUT_CSV_PATH}")
        sys.exit(1)

    # 2. Initial Row Filtering
    logging.info(f"\nStep 2: Dropping rows where '{config.TARGET_COLUMN_FOR_NAN_DROP}' is missing...")
    initial_rows = len(df)
    df.dropna(subset=[config.TARGET_COLUMN_FOR_NAN_DROP], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logging.info(f"  Dropped {initial_rows - len(df)} rows. Current shape: {df.shape}")

    # 3. Drop Low Variation Columns
    logging.info("\nStep 3: Removing columns with low variation...")
    variations = df.nunique()
    cols_to_keep = variations[variations > config.LOW_VARIATION_THRESHOLD].index
    df = df[cols_to_keep]
    logging.info(f"  Shape after dropping low variation columns: {df.shape}")

    # 4. Drop Columns by various criteria
    logging.info("\nStep 4: Dropping columns by keywords, specific names, and older versions...")
    df = drop_columns_by_keywords(df, config.KEYWORDS_TO_DROP)
    df = drop_specific_columns(df, config.SPECIFIC_COLUMNS_TO_DROP)
    df = drop_older_version_columns(df)
    logging.info(f"  Shape after column removal: {df.shape}")

    # 5. Specific Value Replacements
    logging.info("\nStep 5: Performing specific value replacements and type conversions...")
    for column, replacements in config.COLUMNS_FOR_VALUE_REPLACEMENT.items():
        if column in df.columns:
            df[column] = df[column].replace(replacements)
            df[column] = pd.to_numeric(df[column], errors='coerce')

    # 6. Impute Missing Values
    logging.info("\nStep 6: Imputing remaining missing values...")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].replace(r'^\s*$', np.nan, regex=True, inplace=True)
            df[col].fillna('un', inplace=True)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                df[col].fillna(df[col].mean(), inplace=True)
    logging.info("  Imputation complete.")

    # 7. Add Random Feature
    logging.info("\nStep 7: Adding a random feature for modeling robustness checks...")
    df['random_feature'] = np.random.rand(len(df))

    # 8. Save Cleaned Data
    logging.info(f"\nStep 8: Saving cleaned data to {config.CLEANED_CSV_PATH}...")
    df.to_csv(config.CLEANED_CSV_PATH, index=False, encoding='utf-8')
    logging.info(f"  Successfully saved. Final shape: {df.shape}")
    logging.info("--- Data cleaning process finished. ---")


if __name__ == '__main__':
    main()