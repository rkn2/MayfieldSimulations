import pandas as pd
import numpy as np
import logging
import sys
import os
np.random.seed(0)

# --- Logging Configuration Setup ---
# This can be moved to a separate logging_config.py file if you prefer
def setup_logging(log_file='pipeline.log'):
    """Sets up logging to both a file and the console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# Call the setup function to configure logging
setup_logging()


# --- Configuration ---
INPUT_CSV_PATH = 'QuadState_Tornado_DataInputv2.csv'
OUTPUT_CSV_PATH = 'cleaned_data_latlong.csv'

# --- Column Management ---
TARGET_COLUMN_FOR_NAN_DROP = 'degree_of_damage_u' # Initial row filter
LOW_VARIATION_THRESHOLD = 1 # Keep columns with more unique values than this
PERCENT_COLUMNS_ACTION = 'keep' # 'keep' or 'drop'
DIRECTIONAL_COLUMNS_ACTION = 'keep' # 'keep' or 'drop'

# Keywords/patterns for column removal (case-insensitive)
KEYWORDS_TO_DROP = ['photos', 'details', 'prop_', '_unc']

# Specific columns to remove
SPECIFIC_COLUMNS_TO_DROP = [
    'completed_by', 'damage_status', 'ref# (DELETE LATER)', 'complete_address',
    'building_name_listing', 'building_name_current', 'notes', 'tornado_name',
    'tornado_EF', 'tornado_start_lat', 'tornado_start_long', 'tornado_end_lat',
    'tornado_end_long', 'national_register_listing_year', 'town',
    'located_in_historic_district', 'hazards_present_u'
]

# Columns for specific value replacement and potential numeric conversion
COLUMNS_FOR_VALUE_REPLACEMENT = {
    'wall_thickness': {'un': '', 'not_applicable': 0},
    'overhang_length_u': {'un': '', 'not_applicable': 0},
    'parapet_height_m': {'un': '', 'not_applicable': 0}
}

# --- Helper Functions ---

def drop_columns_by_keywords(df, keywords):
    """Drops columns containing any of the specified keywords (case-insensitive)."""
    initial_cols = set(df.columns)
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
    initial_cols = set(df.columns)
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
        base_name = u_col[:-2] # Remove '_u'
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

def handle_percent_columns(df, action):
    """Keeps or drops columns containing '_per_' based on the action."""
    action = action.lower() # Make action case-insensitive

    if action == 'drop':
        cols_to_drop_per = [col for col in df.columns if '_per_' in col]
        if cols_to_drop_per:
            logging.info(f"Action '{action}': Dropping columns containing '_per_': {cols_to_drop_per}")
            df = df.drop(columns=cols_to_drop_per)
            logging.info(f"  Dropped {len(cols_to_drop_per)} columns containing '_per_'.")
        else:
            logging.info(f"Action '{action}': No columns containing '_per_' found to drop.")
    elif action == 'keep':
        logging.info(f"Action '{action}': Keeping all columns containing '_per_'.")
    else:
        logging.warning(f"Invalid PERCENT_COLUMNS_ACTION '{action}'. Defaulting to 'keep'.")

    return df

def handle_directional_columns(df, action):
    """Keeps or drops columns ending with _n, _s, _e, or _w based on the action."""
    action = action.lower() # Make action case-insensitive
    directional_suffixes = ('_n', '_s', '_e', '_w')

    if action == 'drop':
        # Find columns ending with any of the directional suffixes
        cols_to_drop_dir = [col for col in df.columns if col.endswith(directional_suffixes)]
        if cols_to_drop_dir:
            logging.info(f"Action '{action}': Dropping columns ending in {directional_suffixes}: {cols_to_drop_dir}")
            df = df.drop(columns=cols_to_drop_dir)
            logging.info(f"  Dropped {len(cols_to_drop_dir)} directional columns.")
        else:
            logging.info(f"Action '{action}': No directional columns found to drop.")
    elif action == 'keep':
        logging.info(f"Action '{action}': Keeping all directional columns.")
    else:
        logging.warning(f"Invalid DIRECTIONAL_COLUMNS_ACTION '{action}'. Defaulting to 'keep'.")

    return df

# --- Main Data Cleaning Script ---

logging.info(f"Starting data cleaning process for {INPUT_CSV_PATH}")

# 1. Load Data
logging.info("Step 1: Loading data...")
try:
    # Specify low_memory=False if you encounter DtypeWarning, often due to mixed types
    df = pd.read_csv(INPUT_CSV_PATH, delimiter=',', encoding='latin-1', low_memory=False)
    logging.info(f"  Successfully loaded. Initial shape: {df.shape}")
except FileNotFoundError:
    logging.error(f"Error: Input file not found at {INPUT_CSV_PATH}")
    exit() # Stop execution if file is missing
except Exception as e:
    logging.error(f"Error loading CSV: {e}")
    exit()

# 2. Initial Row Filtering
logging.info(f"\nStep 2: Dropping rows where '{TARGET_COLUMN_FOR_NAN_DROP}' is missing...")
initial_rows = len(df)
df = df.dropna(subset=[TARGET_COLUMN_FOR_NAN_DROP])
df = df.reset_index(drop=True)
logging.info(f"  Dropped {initial_rows - len(df)} rows. Current shape: {df.shape}")

# 3. Drop Low Variation Columns
logging.info("\nStep 3: Removing columns with low variation...")
initial_cols = df.shape[1]
variations = df.nunique()
columns_to_keep = variations[variations > LOW_VARIATION_THRESHOLD].index
columns_dropped_variation = set(df.columns) - set(columns_to_keep)
if columns_dropped_variation:
    logging.info(f"  Dropping {len(columns_dropped_variation)} low variation columns: {list(columns_dropped_variation)}")
    df = df[columns_to_keep]
else:
    logging.info("  No low variation columns found to drop.")
logging.info(f"  Shape after dropping low variation columns: {df.shape}")

# 4. Drop Columns by Keywords and Specific Names
logging.info("\nStep 4: Dropping columns by keywords and specific names...")
df = drop_columns_by_keywords(df, KEYWORDS_TO_DROP)
df = drop_specific_columns(df, SPECIFIC_COLUMNS_TO_DROP)
df = handle_percent_columns(df, PERCENT_COLUMNS_ACTION)
df = handle_directional_columns(df, DIRECTIONAL_COLUMNS_ACTION)
logging.info(f"  Shape after dropping keyword/specific columns: {df.shape}")


# 5. Drop Older Version ('_o') Columns
logging.info("\nStep 5: Dropping older version ('_o') columns where updated ('_u') exists...")
df = drop_older_version_columns(df)
logging.info(f"  Shape after dropping older version columns: {df.shape}")

# 6. Specific Value Replacements & Type Conversion
logging.info("\nStep 6: Performing specific value replacements...")
for column, replacements in COLUMNS_FOR_VALUE_REPLACEMENT.items():
    if column in df.columns:
        logging.info(f"  Replacing values in column '{column}': {replacements}")
        df[column] = df[column].replace(replacements)
        # Attempt to convert to numeric after replacements, coercing errors
        # This turns non-numeric values (like remaining strings) into NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        if df[column].isnull().any():
             logging.info(f"    Note: Column '{column}' contains non-numeric values after replacement, converted to NaN.")
    else:
        logging.warning(f"  Column '{column}' not found for value replacement.")

# 7. Impute Missing Values
logging.info("\nStep 7: Imputing missing values...")
numeric_cols_imputed = []
object_cols_imputed = []

for col in df.columns:
    # Convert potential whitespace-only strings to NaN before type check
    # This helps catch cases where a numeric column might have ' ' entries
    if df[col].dtype == 'object':
         df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)

    # Check if column is numeric (or became numeric after replacements)
    if pd.api.types.is_numeric_dtype(df[col]):
        if df[col].isnull().any():
            col_mean = df[col].mean() # NaNs are ignored by default
            df[col] = df[col].fillna(col_mean)
            numeric_cols_imputed.append(f"{col} (mean={col_mean:.2f})")
    # Handle object/string columns
    else:
         # Fill actual NaNs first, then replace any remaining empty strings (shouldn't be many after regex replace)
        if df[col].isnull().any() or (df[col] == '').any():
            df[col] = df[col].fillna('un')
            df[col] = df[col].replace('', 'un') # Catch any residual empty strings
            object_cols_imputed.append(col)

logging.info(f"  Imputed numeric columns with mean: {numeric_cols_imputed if numeric_cols_imputed else 'None'}")
logging.info(f"  Imputed object columns with 'un': {object_cols_imputed if object_cols_imputed else 'None'}")

# 8. Add Random Feature (Optional - kept from original)
logging.info("\nStep 8: Adding a random feature column...")
df['random_feature'] = np.random.rand(len(df))
logging.info("  'random_feature' column added.")

# 9. Save Cleaned Data
logging.info(f"\nStep 9: Saving cleaned data to {OUTPUT_CSV_PATH}...")
try:
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8') # Use utf-8 for broader compatibility
    logging.info(f"  Successfully saved cleaned data. Final shape: {df.shape}")
except Exception as e:
    logging.error(f"Error saving CSV: {e}")

logging.info("\nData cleaning process finished.")
