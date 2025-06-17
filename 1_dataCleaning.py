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
    """Sets up logging to both a file and the console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler (appends to the log file)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Configure logging
setup_logging()


# --- Helper Functions ---

def drop_columns_by_keywords(df, keywords):
    """Drops columns containing any of the specified keywords (case-insensitive)."""
    cols_to_drop = [
        col for col in df.columns
        if any(keyword.lower() in col.lower() for keyword in keywords)
    ]
    if cols_to_drop:
        logging.info(f"Dropping columns based on keywords {keywords}: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        logging.info(f"  Dropped {len(cols_to_drop)} columns.")
    else:
        logging.info(f"No columns found matching keywords {keywords}.")
    return df

def drop_specific_columns(df, columns_list):
    """Drops columns specified in the list if they exist in the DataFrame."""
    cols_to_drop = [col for col in columns_list if col in df.columns]
    cols_not_found = [col for col in columns_list if col not in df.columns]

    if cols_not_found:
        logging.warning(f"Specific columns to drop not found: {cols_not_found}")
    if cols_to_drop:
        logging.info(f"Dropping specific columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        logging.info(f"  Dropped {len(cols_to_drop)} columns.")
    else:
        logging.info("No specific columns found to drop from the provided list.")
    return df

def drop_older_version_columns(df):
    """Drops '_o' columns if a corresponding '_u' column exists."""
    cols_to_drop = []
    updated_cols = {col for col in df.columns if col.endswith('_u')}

    for u_col in updated_cols:
        base_name = u_col[:-2]
        o_col = f"{base_name}_o"
        if o_col in df.columns:
            cols_to_drop.append(o_col)

    if cols_to_drop:
        logging.info(f"Dropping older version columns ('_o' suffix): {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
        logging.info(f"  Dropped {len(cols_to_drop)} columns.")
    else:
        logging.info("No older version ('_o') columns found to drop.")
    return df

# --- Main Data Cleaning Script ---

logging.info(f"--- Starting Script: 1_dataCleaning.py ---")
logging.info(f"Starting data cleaning process for {config.INPUT_CSV_PATH}")

# 1. Load Data
logging.info("Step 1: Loading data...")
try:
    df = pd.read_csv(config.INPUT_CSV_PATH, delimiter=',', encoding='latin-1', low_memory=False)
    logging.info(f"  Successfully loaded. Initial shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Input file not found at {config.INPUT_CSV_PATH}")
    sys.exit()
except Exception as e:
    logging.error(f"Error loading CSV: {e}")
    sys.exit()

# 2. Initial Row Filtering
logging.info(f"\nStep 2: Dropping rows where '{config.TARGET_COLUMN_FOR_NAN_DROP}' is missing...")
initial_rows = len(df)
df = df.dropna(subset=[config.TARGET_COLUMN_FOR_NAN_DROP])
df = df.reset_index(drop=True)
logging.info(f"  Dropped {initial_rows - len(df)} rows. Current shape: {df.shape}")

# 3. Drop Low Variation Columns
logging.info("\nStep 3: Removing columns with low variation...")
variations = df.nunique()
columns_to_keep = variations[variations > config.LOW_VARIATION_THRESHOLD].index
columns_dropped_variation = set(df.columns) - set(columns_to_keep)
if columns_dropped_variation:
    logging.info(f"  Dropping {len(columns_dropped_variation)} low variation columns: {list(columns_dropped_variation)}")
    df = df[columns_to_keep]
else:
    logging.info("  No low variation columns found to drop.")
logging.info(f"  Shape after dropping low variation columns: {df.shape}")

# 4. Drop Columns by Keywords and Specific Names
logging.info("\nStep 4: Dropping columns by keywords and specific names...")
df = drop_columns_by_keywords(df, config.KEYWORDS_TO_DROP)
df = drop_specific_columns(df, config.SPECIFIC_COLUMNS_TO_DROP)
logging.info(f"  Shape after dropping keyword/specific columns: {df.shape}")

# 5. Drop Older Version ('_o') Columns
logging.info("\nStep 5: Dropping older version ('_o') columns where updated ('_u') exists...")
df = drop_older_version_columns(df)
logging.info(f"  Shape after dropping older version columns: {df.shape}")

# 6. Specific Value Replacements & Type Conversion
logging.info("\nStep 6: Performing specific value replacements...")
for column, replacements in config.COLUMNS_FOR_VALUE_REPLACEMENT.items():
    if column in df.columns:
        logging.info(f"  Replacing values in column '{column}': {replacements}")
        df[column] = df[column].replace(replacements)
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isnull().any():
             logging.warning(f"    Note: Column '{column}' contains non-numeric values after replacement, converted to NaN.")
    else:
        logging.warning(f"  Column '{column}' not found for value replacement.")

# 7. Impute Missing Values
logging.info("\nStep 7: Imputing missing values...")
for col in df.columns:
    if df[col].dtype == 'object':
         df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

    if pd.api.types.is_numeric_dtype(df[col]):
        if df[col].isnull().any():
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
    else:
        if df[col].isnull().any() or (df[col] == '').any():
            df[col] = df[col].fillna('un')
            df[col] = df[col].replace('', 'un')

logging.info("  Imputation complete for numeric (mean) and object ('un') columns.")

# 8. Add Random Feature
logging.info("\nStep 8: Adding a random feature column...")
df['random_feature'] = np.random.rand(len(df))
logging.info("  'random_feature' column added.")

# 9. Save Cleaned Data
logging.info(f"\nStep 9: Saving cleaned data to {config.CLEANED_CSV_PATH}...")
try:
    df.to_csv(config.CLEANED_CSV_PATH, index=False, encoding='utf-8')
    logging.info(f"  Successfully saved cleaned data. Final shape: {df.shape}")
except Exception as e:
    logging.error(f"Error saving CSV: {e}")

logging.info("\n--- Data cleaning process finished. ---")