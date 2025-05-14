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

SUBSAMPLE_DAMAGE_0 = None  # Number of rows to keep for damage level 0.  Set to None to disable.

INPUT_CSV_PATH = 'cleaned_data_latlong.csv' # Output from the previous cleaning script
TARGET_COLUMN = 'degree_of_damage_u'

# --- Class Reduction Configuration ---
REDUCE_CLASSES_STRATEGY = 'A' # Options: 'A', 'B', None, etc.
# Define mappings for different strategies
CLASS_MAPPINGS = {
    'A': { # Strategy A: Undamaged, Low-to-Moderate, Significant, Demolished
        0: 0,  # Undamaged
        1: 1,  # Low-to-Moderate Damage
        2: 1,  # Low-to-Moderate Damage
        3: 1,  # Low-to-Moderate Damage
        4: 2,  # Significant Damage
        5: 3   # Demolished
    },
    # You can add more strategies here, e.g., 'B'
    'B': { # Example Strategy B: Undamaged, Damaged-Repairable, Demolished
        0: 0,  # Undamaged
        1: 1,  # Damaged-Repairable
        2: 1,  # Damaged-Repairable
        3: 1,  # Damaged-Repairable
        4: 1,  # Damaged-Repairable
        5: 2   # Demolished
    }
}
# Define new category order for plotting if classes are reduced
# This needs to be adjusted based on the CHOSEN strategy
NEW_CATEGORY_ORDER = { # Maps strategy to its category order
    'A': [0, 1, 2, 3],
    'B': [0, 1, 2]
}
NEW_CATEGORY_LABELS = { # Maps strategy to its new labels
    'A': ['Undamaged (0)', 'Low-Mod (1)', 'Significant (2)', 'Demolished (3)'],
    'B': ['Undamaged (0)', 'Repairable (1)', 'Demolished (2)']
}


# --- Balancing Configuration ---
BALANCING_METHOD = 'RandomOverSampler'
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
    BALANCING_METHOD = None

def filter_features(df, target_col, keywords_to_remove):
    cols_to_drop = set() # Start with empty set, target col handled elsewhere
    keywords_lower = [kw.lower() for kw in keywords_to_remove]
    for col in df.columns:
        if col == target_col: # Skip the target column itself in this loop
            continue
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in keywords_lower):
            cols_to_drop.add(col)
    cols_to_drop_existing = [col for col in cols_to_drop if col in df.columns]
    print(f"Filtering features. Removing columns based on keywords (target column already separated):")
    print(f"  Keywords: {keywords_to_remove}")
    print(f"  Columns identified for removal from X: {sorted(list(cols_to_drop_existing))}")
    X = df.drop(columns=cols_to_drop_existing, errors='ignore')
    print(f"  Number of features remaining in X: {X.shape[1]}")
    return X

print(f"Starting ML preprocessing for {INPUT_CSV_PATH}")
print(f"Selected Class Reduction Strategy: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'}")
print(f"Selected Balancing Method: {BALANCING_METHOD if BALANCING_METHOD else 'None'}")
if BALANCING_METHOD:
    print(f"Sampling Strategy: {SAMPLING_STRATEGY}")

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

if SUBSAMPLE_DAMAGE_0 is not None and (REDUCE_CLASSES_STRATEGY is None or (REDUCE_CLASSES_STRATEGY and CLASS_MAPPINGS.get(REDUCE_CLASSES_STRATEGY, {}).get(0) == 0)):
    # Only subsample original class 0 if it remains class 0 after mapping OR if no mapping is done
    print(f"\nStep 2: Subsampling original damage level 0 to {SUBSAMPLE_DAMAGE_0} rows...")
    damage_0_indices = df[df[TARGET_COLUMN] == 0].index
    if len(damage_0_indices) > SUBSAMPLE_DAMAGE_0:
        random_indices = np.random.choice(damage_0_indices, SUBSAMPLE_DAMAGE_0, replace=False)
        other_indices = df.index.difference(damage_0_indices) # Get all other indices
        # Concatenate the chosen damage 0 indices with all other indices
        df = df.loc[pd.Index(random_indices).union(other_indices)]
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) # Shuffle
        print(f"  Subsampled DataFrame shape: {df.shape}")
    else:
        print(f"  Original damage level 0 has {len(damage_0_indices)} rows, which is <= {SUBSAMPLE_DAMAGE_0}. Skipping subsampling.")
elif SUBSAMPLE_DAMAGE_0 is not None:
    print(f"\nStep 2: Subsampling of original damage level 0 skipped because it's being remapped by strategy {REDUCE_CLASSES_STRATEGY}.")


print(f"\nStep 3: Separating target variable '{TARGET_COLUMN}'...")
if TARGET_COLUMN not in df.columns:
    print(f"Error: Target column '{TARGET_COLUMN}' not found in the loaded data.")
    exit()
y_original = df[TARGET_COLUMN] # Keep a copy of original y for initial visualization
try:
    y = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').fillna(0).astype(int) # Use df[TARGET_COLUMN] to get a fresh copy
    print(f"  Target variable '{TARGET_COLUMN}' separated and converted to integer type.")
    print(f"  Original y distribution (after any subsampling):\n{y.value_counts(normalize=True).sort_index()}")
except Exception as e:
    print(f"Warning: Could not convert target variable '{TARGET_COLUMN}' to integer. Error: {e}")
    print(f"  Keeping original type: {y.dtype}")
    y_original = y.copy() # Ensure y_original has the same data as y

# --- NEW STEP: Apply Class Reduction ---
print("\nStep 3.5: Applying Class Reduction...")
if REDUCE_CLASSES_STRATEGY and REDUCE_CLASSES_STRATEGY in CLASS_MAPPINGS:
    mapping_to_apply = CLASS_MAPPINGS[REDUCE_CLASSES_STRATEGY]
    y_reduced = y.map(mapping_to_apply)
    print(f"  Applied class reduction strategy '{REDUCE_CLASSES_STRATEGY}'.")
    print(f"  y distribution BEFORE reduction (from current df):\n{y.value_counts(normalize=True).sort_index()}")
    print(f"  y distribution AFTER reduction:\n{y_reduced.value_counts(normalize=True).sort_index()}")
    y = y_reduced # Overwrite y with the reduced classes
    # Adjust category order and labels for plotting
    current_category_order = NEW_CATEGORY_ORDER.get(REDUCE_CLASSES_STRATEGY, sorted(y.unique()))
    current_category_labels = NEW_CATEGORY_LABELS.get(REDUCE_CLASSES_STRATEGY, [str(i) for i in current_category_order])
else:
    print("  No class reduction strategy applied or strategy not found.")
    current_category_order = sorted(y.unique()) # Default order for original classes
    current_category_labels = [str(i) for i in current_category_order]


# Step 4: Filter Feature Set (X) - now pass df.drop(TARGET_COLUMN) or just X=df and filter_features handles it
print("\nStep 4: Filtering feature set (X)...")
X = filter_features(df.drop(columns=[TARGET_COLUMN]), TARGET_COLUMN, KEYWORDS_TO_REMOVE_FROM_X) # Pass X without target
print(f"  Shape of initial feature set X: {X.shape}")

print("\nStep 5: Identifying numeric and categorical features in the filtered X...")
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()
all_features = numeric_features + categorical_features
if len(all_features) != X.shape[1]:
    warnings.warn("Mismatch between identified features and total columns in X.")
print(f"  Identified {len(numeric_features)} numeric features.")
print(f"  Identified {len(categorical_features)} categorical features.")

print("\nStep 6: Defining preprocessing steps...")
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for easier DataFrame conversion
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)
print("  Preprocessor defined (StandardScaler for numeric, OneHotEncoder for categorical).")

print("\nStep 7: Splitting data into training and testing sets...")
# y here is ALREADY REDUCED if strategy was applied
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y # Stratify on the (potentially reduced) y
    )
    print(f"  Data split successfully (stratified on {'reduced' if REDUCE_CLASSES_STRATEGY else 'original'} classes):")
except ValueError as e:
     print(f"Warning: Could not stratify split (error: {e}). Splitting without stratification.")
     X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
     print(f"  Data split successfully (without stratification):")

print(f"    X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"    X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
print(f"    y_train distribution (after reduction, before balancing):\n{y_train.value_counts(normalize=True).sort_index()}")
print(f"    y_test distribution (after reduction):\n{y_test.value_counts(normalize=True).sort_index()}")

print("\nStep 8: Applying preprocessing (fitting on train, transforming train and test)...")
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names_out = None
try:
    feature_names_out = preprocessor.get_feature_names_out()
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names_out, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_out, index=X_test.index)
    print("  Preprocessing applied. Converted processed arrays to DataFrames.")
except Exception as e:
    print(f"Warning: Could not get feature names or create DataFrames from processed arrays. Error: {e}")
    print("  Proceeding with NumPy arrays for processed X data.")
    X_train_processed_df = X_train_processed
    X_test_processed_df = X_test_processed

print(f"  Shape after preprocessing:")
print(f"    X_train_processed shape: {X_train_processed_df.shape}")
print(f"    X_test_processed shape:  {X_test_processed_df.shape}")

X_train_final = X_train_processed_df # Initialize final variables
y_train_final = y_train

if BALANCING_METHOD and sampler_class:
    print(f"  Balancing is ENABLED using {BALANCING_METHOD} (strategy='{SAMPLING_STRATEGY}')...")
    try:
        sampler = sampler_class(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
        print(f"  Instantiated {BALANCING_METHOD} sampler.")
    except Exception as e:
        print(f"Error instantiating sampler {BALANCING_METHOD}: {e}")
        print("  Skipping balancing.")
        BALANCING_METHOD = None # Disable balancing if instantiation fails

    if BALANCING_METHOD:
        try:
            print(f"  Applying {BALANCING_METHOD}.fit_resample to training data...")
            # Ensure X_train_processed_df is numeric for SMOTE if it's used
            if BALANCING_METHOD == 'SMOTE' and not isinstance(X_train_processed_df, np.ndarray):
                 # SMOTE expects numpy array
                 X_train_to_resample = X_train_processed_df.to_numpy() if hasattr(X_train_processed_df, 'to_numpy') else X_train_processed_df
            else:
                 X_train_to_resample = X_train_processed_df

            X_train_resampled_np, y_train_resampled_np = sampler.fit_resample(X_train_to_resample, y_train)
            print(f"  Resampling complete.")
            print(f"    Original y_train distribution (after reduction):\n{y_train.value_counts().sort_index()}")
            y_train_resampled_series = pd.Series(y_train_resampled_np, name=y_train.name)
            print(f"    Resampled y_train distribution:\n{y_train_resampled_series.value_counts().sort_index()}")
            print(f"    X_train_resampled shape: {X_train_resampled_np.shape}")

            y_train_final = y_train_resampled_series

            if isinstance(X_train_processed_df, pd.DataFrame) and feature_names_out is not None:
                 try:
                     X_train_final = pd.DataFrame(X_train_resampled_np, columns=feature_names_out)
                     print("  Converted resampled X_train back to DataFrame.")
                 except Exception as e:
                     print(f"Warning: Could not convert resampled X_train back to DataFrame: {e}. Keeping as NumPy array.")
                     X_train_final = X_train_resampled_np
            else:
                X_train_final = X_train_resampled_np
        except Exception as e:
            print(f"Error during {BALANCING_METHOD} fit_resample: {e}")
            print("  Balancing failed. Using original processed training data (after reduction).")
            X_train_final = X_train_processed_df # Already has processed, non-resampled data
            y_train_final = y_train # Already has y_train (reduced, non-resampled)
            BALANCING_METHOD = None
else:
    if BALANCING_METHOD: # Only print if a method was *intended* but failed earlier
         print(f"  Balancing method '{BALANCING_METHOD}' was specified but could not be applied.")
    print("  Using training data (after reduction, no balancing applied).")
    # X_train_final and y_train_final are already set to X_train_processed_df and y_train

print("\nStep 9: Saving final data...")
if SAVE_PROCESSED_DATA:
    print(f"  Saving processed data and preprocessor to '{OUTPUT_DIR}'...")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        joblib.dump(X_train_final, OUTPUT_TRAIN_X_PATH)
        joblib.dump(y_train_final, OUTPUT_TRAIN_Y_PATH)
        joblib.dump(X_test_processed_df, OUTPUT_TEST_X_PATH)
        joblib.dump(y_test, OUTPUT_TEST_Y_PATH) # y_test is also reduced
        joblib.dump(preprocessor, OUTPUT_PREPROCESSOR_PATH)
        print(f"  Successfully saved:")
        applied_balancing_status = BALANCING_METHOD if BALANCING_METHOD and 'y_train_resampled_series' in locals() and y_train_final.shape == y_train_resampled_series.shape else 'None'
        print(f"    - Train X: {OUTPUT_TRAIN_X_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'}, Balancing: {applied_balancing_status})")
        print(f"    - Train y: {OUTPUT_TRAIN_Y_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'}, Balancing: {applied_balancing_status})")
        print(f"    - Test X:  {OUTPUT_TEST_X_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'})")
        print(f"    - Test y:  {OUTPUT_TEST_Y_PATH} (Class Reduction: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else 'None'})")
        print(f"    - Preprocessor: {OUTPUT_PREPROCESSOR_PATH}")
    except Exception as e:
        print(f"Error saving processed data: {e}")
else:
    print("  Skipping saving processed data (SAVE_PROCESSED_DATA is False).")

print("\nML preprocessing finished.")

# --- Visualization Code (Subplots) ---
print("\n--- Starting Visualization (Subplots) ---")

fig, axes = plt.subplots(2, 2, figsize=(18, 14)) # Adjusted figsize
fig.suptitle(f'Distribution of Degree of Damage (Strategy: {REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else "Original"})', fontsize=16)

# Plotting y_original (before any reduction or subsampling related to class 0 for this specific plot)
# To get a true original distribution, reload y from the very first df load if needed,
# or use the y_original we saved right after loading df[TARGET_COLUMN]
# For simplicity here, using y_original as it was when separated.
original_plot_y = pd.read_csv(INPUT_CSV_PATH, low_memory=False)[TARGET_COLUMN].astype(int)
sns.countplot(x=original_plot_y, palette='viridis', ax=axes[0, 0], order=list(range(6))) # Plot original 0-5
axes[0, 0].set_title('True Original Data (Before Any Processing)')
axes[0, 0].set_xlabel('Degree of Damage')
axes[0, 0].set_ylabel('Number of Occurrences')
axes[0, 0].set_xticks(ticks=range(6), labels=[str(i) for i in range(6)])
axes[0, 0].set_xlim(-0.5, 5.5)


# y_train distribution (this is after class reduction, before balancing)
sns.countplot(x=y_train, palette='viridis', ax=axes[0, 1], order=current_category_order)
axes[0, 1].set_title('Training Data (After Reduction, Before Balancing)')
axes[0, 1].set_xlabel('Degree of Damage')
axes[0, 1].set_ylabel('Number of Occurrences')
axes[0, 1].set_xticks(ticks=current_category_order, labels=current_category_labels)
axes[0, 1].tick_params(axis='x', rotation=45) # Rotate labels if they overlap
axes[0, 1].set_xlim(min(current_category_order)-0.5, max(current_category_order)+0.5)


# y_train_final distribution (this is after class reduction AND after balancing)
sns.countplot(x=y_train_final, palette='viridis', ax=axes[1, 0], order=current_category_order)
axes[1, 0].set_title(f'Training Data (After Reduction & Balancing: {applied_balancing_status})')
axes[1, 0].set_xlabel('Degree of Damage')
axes[1, 0].set_ylabel('Number of Occurrences')
axes[1, 0].set_xticks(ticks=current_category_order, labels=current_category_labels)
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].set_xlim(min(current_category_order)-0.5, max(current_category_order)+0.5)

# y_test distribution (this is after class reduction)
sns.countplot(x=y_test, palette='viridis', ax=axes[1, 1], order=current_category_order)
axes[1, 1].set_title('Test Data (After Reduction)')
axes[1, 1].set_xlabel('Degree of Damage')
axes[1, 1].set_ylabel('Number of Occurrences')
axes[1, 1].set_xticks(ticks=current_category_order, labels=current_category_labels)
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].set_xlim(min(current_category_order)-0.5, max(current_category_order)+0.5)


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_plot_filename = f'degree_of_damage_subplots_strategy_{REDUCE_CLASSES_STRATEGY if REDUCE_CLASSES_STRATEGY else "orig"}.png'
plt.savefig(output_plot_filename)
print(f"Saved plot to {output_plot_filename}")
plt.show()