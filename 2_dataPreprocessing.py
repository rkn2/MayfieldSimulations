import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import os
import joblib # For saving/loading sklearn objects and data efficiently
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---

SUBSAMPLE_DAMAGE_0 = 25  # Number of rows to keep for damage level 0.  Set to None to disable.

INPUT_CSV_PATH = 'cleaned_data_latlong.csv' # Output from the previous cleaning script
TARGET_COLUMN = 'degree_of_damage_u'
# BALANCING_ENABLED = True # Set BALANCING_ENABLED to True to apply Random Oversampling to the training data
# BALANCING_STRATEGY = 'auto' # Strategy for RandomOverSampler, 'auto' balances all minority classes to match majority

# --- Balancing Configuration ---
# Choose the balancing method: 'RandomOverSampler', 'SMOTE', or None
BALANCING_METHOD = 'RandomOverSampler' # Options: 'RandomOverSampler', 'SMOTE', None
# Strategy for the chosen sampler. 'auto' typically balances all minority classes.
# For SMOTE, 'auto' is equivalent to 'not majority'. Other options exist like 'minority', 'not majority', 'all'.
# For RandomOverSampler, 'auto' is equivalent to 'not majority'.
SAMPLING_STRATEGY = 'auto'
# Note: SMOTE has other parameters like k_neighbors, which we'll leave as default for now.


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
sampler_class = None
if BALANCING_METHOD in ['RandomOverSampler', 'SMOTE']:
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        if BALANCING_METHOD == 'RandomOverSampler':
            sampler_class = RandomOverSampler
            print("Selected balancing method: RandomOverSampler")
        elif BALANCING_METHOD == 'SMOTE':
            sampler_class = SMOTE
            print("Selected balancing method: SMOTE")
    except ImportError:
        print(f"Error: 'imbalanced-learn' library not found, but BALANCING_METHOD is set to '{BALANCING_METHOD}'.")
        print("Please install it: pip install imbalanced-learn")
        exit()
elif BALANCING_METHOD is not None:
    print(f"Warning: Unknown BALANCING_METHOD '{BALANCING_METHOD}'. Balancing will be disabled.")
    BALANCING_METHOD = None # Disable balancing if method is unknown

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
print(f"Selected Balancing Method: {BALANCING_METHOD if BALANCING_METHOD else 'None'}")
if BALANCING_METHOD:
    print(f"Sampling Strategy: {SAMPLING_STRATEGY}")

# 1. Load Cleaned Data
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

# 2. Subsample Damage Level 0 (Before Splitting)
if SUBSAMPLE_DAMAGE_0 is not None:
    print(f"\nStep 2: Subsampling damage level 0 to {SUBSAMPLE_DAMAGE_0} rows...")
    damage_0_indices = df[df[TARGET_COLUMN] == 0].index
    if len(damage_0_indices) > SUBSAMPLE_DAMAGE_0:
        random_indices = np.random.choice(damage_0_indices, SUBSAMPLE_DAMAGE_0, replace=False)
        other_indices = df.index.difference(damage_0_indices)
        df = df.loc[np.concatenate([random_indices, other_indices])]
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) # Shuffle the dataframe
        print(f"  Subsampled DataFrame shape: {df.shape}")
    else:
        print(f"  Damage level 0 has {len(damage_0_indices)} rows, which is <= SUBSAMPLE_DAMAGE_0. Skipping subsampling.")


# 3. Separate Target Variable (y)
# ... (same as before) ...
print(f"\nStep 3: Separating target variable '{TARGET_COLUMN}'...")
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

# 4. Filter Feature Set (X)
# ... (same as before) ...
print("\nStep 4: Filtering feature set (X)...")
X = filter_features(df, TARGET_COLUMN, KEYWORDS_TO_REMOVE_FROM_X)
print(f"  Shape of initial feature set X: {X.shape}")


# 5. Identify Feature Types in Filtered X
# ... (same as before) ...
print("\nStep 5: Identifying numeric and categorical features in the filtered X...")
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
all_features = numeric_features + categorical_features
if len(all_features) != X.shape[1]:
    warnings.warn("Mismatch between identified features and total columns in X.")
print(f"  Identified {len(numeric_features)} numeric features.")
print(f"  Identified {len(categorical_features)} categorical features.")


# 6. Define Preprocessing Steps
# ... (same as before) ...
print("\nStep 6: Defining preprocessing steps...")
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


# 7. Split Data into Training and Testing Sets
# ... (same as before) ...
print("\nStep 7: Splitting data into training and testing sets...")
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


# 8. Apply Preprocessing
print("\nStep 8: Applying preprocessing (fitting on train, transforming train and test)...")
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


# Check if a balancing method was chosen and the sampler class was successfully imported
if BALANCING_METHOD and sampler_class:
    print(f"  Balancing is ENABLED using {BALANCING_METHOD} (strategy='{SAMPLING_STRATEGY}')...")
    # Instantiate the selected sampler
    # Note: SMOTE might require numeric data only. Our preprocessor converts categories, so this should be okay.
    #       If using sparse matrices from OneHotEncoder, check sampler compatibility.
    try:
        sampler = sampler_class(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
        print(f"  Instantiated {BALANCING_METHOD} sampler.")
    except Exception as e:
        print(f"Error instantiating sampler {BALANCING_METHOD}: {e}")
        print("  Skipping balancing.")
        BALANCING_METHOD = None # Disable balancing if instantiation fails

    if BALANCING_METHOD: # Check again in case instantiation failed
        try:
            # Apply resampling ONLY to the processed training data
            # Ensure y_train is in the correct format (usually a Series or 1D array)
            print(f"  Applying {BALANCING_METHOD}.fit_resample to training data...")
            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_processed_df, y_train)
            print(f"  Resampling complete.")
            print(f"    Original y_train distribution:\n{y_train.value_counts().sort_index()}")
            # Convert resampled y to Series for value_counts
            y_train_resampled_series = pd.Series(y_train_resampled)
            print(f"    Resampled y_train distribution:\n{y_train_resampled_series.value_counts().sort_index()}")
            print(f"    X_train_resampled shape: {X_train_resampled.shape}")

            # Update final variables to the resampled data
            y_train_final = y_train_resampled_series # Use the Series version

            # Convert resampled X back to DataFrame if possible and original was DataFrame
            if isinstance(X_train_processed_df, pd.DataFrame) and feature_names_out is not None:
                 try:
                     # Create DataFrame with original column names, index will be reset
                     X_train_final = pd.DataFrame(X_train_resampled, columns=feature_names_out)
                     print("  Converted resampled X_train back to DataFrame.")
                 except Exception as e:
                     print(f"Warning: Could not convert resampled X_train back to DataFrame: {e}. Keeping as NumPy array.")
                     X_train_final = X_train_resampled # Keep as numpy array
            else:
                X_train_final = X_train_resampled # Keep as numpy array if original wasn't DataFrame or names unavailable

        except Exception as e:
            print(f"Error during {BALANCING_METHOD} fit_resample: {e}")
            print("  Balancing failed. Using original processed training data.")
            # Reset final variables to original processed data if resampling fails
            X_train_final = X_train_processed_df
            y_train_final = y_train
            BALANCING_METHOD = None # Mark balancing as failed/disabled

else:
    # This block executes if BALANCING_METHOD was None initially, or if sampler_class failed to import,
    # or if sampler instantiation failed, or if fit_resample failed.
    if BALANCING_METHOD: # Only print if a method was *intended* but failed earlier
         print(f"  Balancing method '{BALANCING_METHOD}' was specified but could not be applied.")
    print("  Using original processed training data (no balancing applied).")
    # X_train_final and y_train_final already hold the correct (original processed) data

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
        # Indicate which balancing method was actually applied (or None)
        applied_balancing = BALANCING_METHOD if 'y_train_resampled_series' in locals() else 'None'
        print(f"    - Train X: {OUTPUT_TRAIN_X_PATH} (Balancing Applied: {applied_balancing})")
        print(f"    - Train y: {OUTPUT_TRAIN_Y_PATH} (Balancing Applied: {applied_balancing})")
        print(f"    - Test X:  {OUTPUT_TEST_X_PATH}")
        print(f"    - Test y:  {OUTPUT_TEST_Y_PATH}")
        print(f"    - Preprocessor: {OUTPUT_PREPROCESSOR_PATH}")

    except Exception as e:
        print(f"Error saving processed data: {e}")
else:
    print("  Skipping saving processed data (SAVE_PROCESSED_DATA is False).")


print("\nML preprocessing finished.")

# --- Visualization Code (Subplots) ---
print("\n--- Starting Visualization (Subplots) ---")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Create a 2x2 grid of subplots
fig.suptitle('Distribution of Degree of Damage', fontsize=16)

# Define the order of categories for the countplots
category_order = list(range(6))  # [0, 1, 2, 3, 4, 5]

# 1. Original Data
sns.countplot(x=y, palette='viridis', ax=axes[0, 0], order=category_order)
axes[0, 0].set_title('Original Data')
axes[0, 0].set_xlabel('Degree of Damage')
axes[0, 0].set_ylabel('Number of Occurrences')
axes[0, 0].set_xticks(ticks=range(6), labels=[0, 1, 2, 3, 4, 5])
axes[0, 0].set_xlim(-0.5, 5.5)  # Force x-axis limits to include all categories

# 2. Training Data (Before Balancing)
sns.countplot(x=y_train, palette='viridis', ax=axes[0, 1], order=category_order)
axes[0, 1].set_title('Training Data (Before Balancing)')
axes[0, 1].set_xlabel('Degree of Damage')
axes[0, 1].set_ylabel('Number of Occurrences')
axes[0, 1].set_xticks(ticks=range(6), labels=[0, 1, 2, 3, 4, 5])
axes[0, 1].set_xlim(-0.5, 5.5)  # Force x-axis limits to include all categories

# 3. Training Data (After Balancing) - Conditional
if 'y_train_resampled_series' in locals():
    sns.countplot(x=y_train_resampled_series, palette='viridis', ax=axes[1, 0], order=category_order)
    axes[1, 0].set_title('Training Data (After Balancing)')
    axes[1, 0].set_xlabel('Degree of Damage')
    axes[1, 0].set_ylabel('Number of Occurrences')
    axes[1, 0].set_xticks(ticks=range(6), labels=[0, 1, 2, 3, 4, 5])
    axes[1, 0].set_xlim(-0.5, 5.5)  # Force x-axis limits to include all categories
else:
    axes[1, 0].text(0.5, 0.5, 'No balancing applied', ha='center', va='center')
    axes[1, 0].axis('off')  # Turn off the axes if no plot is drawn

# 4. Test Data
sns.countplot(x=y_test, palette='viridis', ax=axes[1, 1], order=category_order)
axes[1, 1].set_title('Test Data')
axes[1, 1].set_xlabel('Degree of Damage')
axes[1, 1].set_ylabel('Number of Occurrences')
axes[1, 1].set_xticks(ticks=range(6), labels=[0, 1, 2, 3, 4, 5])
axes[1, 1].set_xlim(-0.5, 5.5)  # Force x-axis limits to include all categories

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent overlapping titles
plt.savefig('degree_of_damage_subplots.png')
plt.show()