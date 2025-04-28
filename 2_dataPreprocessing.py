import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import os
import joblib # For saving/loading sklearn objects and data efficiently

# --- Configuration ---
INPUT_CSV_PATH = 'cleaned_data_latlong.csv' # Output from the previous cleaning script
TARGET_COLUMN = 'degree_of_damage_u'
BALANCING_ENABLED = False # Set BALANCING_ENABLED to True to apply Random Oversampling to the training data
BALANCING_STRATEGY = 'auto' # Strategy for RandomOverSampler, 'auto' balances all minority classes to match majority

# Keywords to identify feature columns to REMOVE from X (case-insensitive)
KEYWORDS_TO_REMOVE_FROM_X = [
    'damage', 'status_u', 'exist', 'demolish', 'failure', 'after'
]

TEST_SIZE = 0.2
RANDOM_STATE = 42 # Ensures reproducibility of split and sampling



# --- Output Configuration ---
SAVE_PROCESSED_DATA = True
OUTPUT_DIR = 'processed_ml_data'
# File names remain the same, but content of train files will differ based on BALANCING_ENABLED
OUTPUT_TRAIN_X_PATH = f'{OUTPUT_DIR}/X_train_processed.pkl'
OUTPUT_TEST_X_PATH = f'{OUTPUT_DIR}/X_test_processed.pkl'
OUTPUT_TRAIN_Y_PATH = f'{OUTPUT_DIR}/y_train.pkl'
OUTPUT_TEST_Y_PATH = f'{OUTPUT_DIR}/y_test.pkl'
OUTPUT_PREPROCESSOR_PATH = f'{OUTPUT_DIR}/preprocessor.pkl'

# --- Optional: Import imbalanced-learn ---
if BALANCING_ENABLED:
    try:
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        print("Error: 'imbalanced-learn' library not found, but BALANCING_ENABLED is True.")
        print("Please install it: pip install imbalanced-learn")
        exit()

# --- Helper Functions ---
# (filter_features function remains the same as before)
def filter_features(df, target_col, keywords_to_remove):
    """Removes the target column and columns containing specified keywords."""
    cols_to_drop = set([target_col]) # Start with the target column
    keywords_lower = [kw.lower() for kw in keywords_to_remove]

    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords_lower):
            cols_to_drop.add(col)

    # Ensure we don't accidentally try to drop columns that aren't present
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]

    if target_col not in df.columns:
         print(f"Warning: Target column '{target_col}' not found in DataFrame columns.")

    print(f"Filtering features. Removing columns based on target and keywords:")
    print(f"  Keywords: {keywords_to_remove}")
    print(f"  Columns identified for removal: {sorted(list(cols_to_drop_existing))}")

    X = df.drop(columns=cols_to_drop_existing, errors='ignore')
    print(f"  Number of features remaining in X: {X.shape[1]}")
    return X

# --- Main Preprocessing Script ---

print(f"Starting ML preprocessing for {INPUT_CSV_PATH}")
print(f"Balancing Enabled: {BALANCING_ENABLED}")

# 1. Load Cleaned Data
# ... (same as before) ...
print("\nStep 1: Loading cleaned data...")
try:
    df = pd.read_csv(INPUT_CSV_PATH, low_memory=False)
    print(f"  Successfully loaded. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: Cleaned data file not found at {INPUT_CSV_PATH}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# 2. Separate Target Variable (y)
# ... (same as before) ...
print(f"\nStep 2: Separating target variable '{TARGET_COLUMN}'...")
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the loaded data.")
    exit()
y = df[TARGET_COLUMN]
try:
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    print(f"  Target variable '{TARGET_COLUMN}' separated and converted to integer type.")
    print(f"  Original full dataset y distribution:\n{y.value_counts(normalize=True).sort_index()}")
except Exception as e:
    print(f"Warning: Could not convert target variable '{TARGET_COLUMN}' to integer. Error: {e}")
    print(f"  Keeping original type: {y.dtype}")

# 3. Filter Feature Set (X)
# ... (same as before) ...
print("\nStep 3: Filtering feature set (X)...")
X = filter_features(df, TARGET_COLUMN, KEYWORDS_TO_REMOVE_FROM_X)
print(f"  Shape of initial feature set X: {X.shape}")


# 4. Identify Feature Types in Filtered X
# ... (same as before) ...
print("\nStep 4: Identifying numeric and categorical features in the filtered X...")
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
all_features = numeric_features + categorical_features
if len(all_features) != X.shape[1]:
    warnings.warn("Mismatch between identified features and total columns in X.")
print(f"  Identified {len(numeric_features)} numeric features.")
print(f"  Identified {len(categorical_features)} categorical features.")


# 5. Define Preprocessing Steps
# ... (same as before) ...
print("\nStep 5: Defining preprocessing steps...")
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
print("  Preprocessor defined (StandardScaler for numeric, OneHotEncoder for categorical).")


# 6. Split Data into Training and Testing Sets
# ... (same as before) ...
print("\nStep 6: Splitting data into training and testing sets...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Data split successfully (stratified):")
except ValueError as e:
     print(f"Warning: Could not stratify split (error: {e}). Splitting without stratification.")
     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
     print(f"  Data split successfully (without stratification):")

print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"    X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
print(f"    y_train distribution before processing:\n{y_train.value_counts(normalize=True).sort_index()}")
print(f"    y_test distribution:\n{y_test.value_counts(normalize=True).sort_index()}")


# 7. Apply Preprocessing
print("\nStep 7: Applying preprocessing (fitting on train, transforming train and test)...")
# Fit ONLy on X_train
preprocessor.fit(X_train)
# Transform both
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Attempt to get feature names and create DataFrames
feature_names_out = None
try:
    feature_names_out = preprocessor.get_feature_names_out()
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
    print("  Preprocessing applied. Converted processed arrays to DataFrames.")
except Exception as e:
    print(f"Warning: Could not get feature names or create DataFrames from processed arrays. Error: {e}")
    print("  Proceeding with NumPy arrays for processed X data.")
    X_train_processed_df = X_train_processed # Keep as numpy array
    X_test_processed_df = X_test_processed   # Keep as numpy array

print(f"  Shape after preprocessing:")
print(f"    X_train_processed shape: {X_train_processed_df.shape}")
print(f"    X_test_processed shape:  {X_test_processed_df.shape}")


# 8. Apply Balancing (Optional - only to training data)
print("\nStep 8: Applying Balancing (if enabled)...")

# Define final variables to hold the training data (either original processed or resampled)
X_train_final = X_train_processed_df
y_train_final = y_train

if BALANCING_ENABLED:
    print(f"  Balancing is ENABLED. Applying RandomOversampler (strategy='{BALANCING_STRATEGY}')...")
    ros = RandomOverSampler(sampling_strategy=BALANCING_STRATEGY, random_state=RANDOM_STATE)

    # Apply resampling ONLY to the processed training data
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train_processed_df, y_train)

    print(f"  Oversampling complete.")
    print(f"    Original y_train distribution:\n{y_train.value_counts()}")
    print(f"    Resampled y_train distribution:\n{pd.Series(y_train_resampled).value_counts()}")
    print(f"    X_train_resampled shape: {X_train_resampled.shape}")

    # Update final variables to the resampled data
    y_train_final = y_train_resampled
    # Convert resampled X back to DataFrame if possible and needed
    if isinstance(X_train_processed_df, pd.DataFrame) and feature_names_out is not None:
         X_train_final = pd.DataFrame(X_train_resampled, columns=feature_names_out) # Index is lost here, add if needed
    else:
        X_train_final = X_train_resampled # Keep as numpy array

else:
    print("  Balancing is DISABLED. Using original processed training data.")
    # X_train_final and y_train_final already hold the correct data


# 9. Save Processed Data (Optional)
print("\nStep 9: Saving final data...")
if SAVE_PROCESSED_DATA:
    print(f"  Saving processed data and preprocessor to '{OUTPUT_DIR}'...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save the FINAL training data (resampled if BALANCING_ENABLED, else original processed)
        joblib.dump(X_train_final, OUTPUT_TRAIN_X_PATH)
        joblib.dump(y_train_final, OUTPUT_TRAIN_Y_PATH)

        # Save the ORIGINAL processed test data (never resampled)
        joblib.dump(X_test_processed_df, OUTPUT_TEST_X_PATH)
        joblib.dump(y_test, OUTPUT_TEST_Y_PATH)

        # Save the fitted preprocessor (fitted on original X_train)
        joblib.dump(preprocessor, OUTPUT_PREPROCESSOR_PATH)

        print(f"  Successfully saved:")
        print(f"    - Train X: {OUTPUT_TRAIN_X_PATH} (Resampled: {BALANCING_ENABLED})")
        print(f"    - Train y: {OUTPUT_TRAIN_Y_PATH} (Resampled: {BALANCING_ENABLED})")
        print(f"    - Test X:  {OUTPUT_TEST_X_PATH}")
        print(f"    - Test y:  {OUTPUT_TEST_Y_PATH}")
        print(f"    - Preprocessor: {OUTPUT_PREPROCESSOR_PATH}")

    except Exception as e:
        print(f"Error saving processed data: {e}")
else:
    print("  Skipping saving processed data (SAVE_PROCESSED_DATA is False).")


print("\nML preprocessing finished.")