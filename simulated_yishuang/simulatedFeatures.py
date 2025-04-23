
import pandas as pd
import numpy as np
# Replace 'your_file.csv' with the actual path to your file
# Replace ',' with the correct delimiter if it's not a comma
df = pd.read_csv('feature_importance_sims.csv', delimiter=',', encoding='latin-1')
#
## REMOVING COLUMNS
## VARIATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate the variation for each column (you can use other methods like standard deviation)
variations = df.nunique()

# Set a threshold for variation (e.g., columns with only one unique value are removed)
threshold = 1

# Filter out columns with variation below the threshold
columns_to_keep = variations[variations > threshold].index
print(len(df.columns))
print(len(columns_to_keep))
df = df[columns_to_keep]

## SPECIFIC NAMES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Remove columns containing "photos" in their name (case-insensitive)
columns_to_drop = [col for col in df.columns if 'photos' in col.lower()]
df = df.drop(columns=columns_to_drop)

# Remove columns containing "details" in their name (case-insensitive)
columns_to_drop = [col for col in df.columns if 'details' in col.lower()]
df = df.drop(columns=columns_to_drop)

# Remove columns containing "details" in their name (case-insensitive)
columns_to_drop = [col for col in df.columns if 'prop_' in col.lower()]
df = df.drop(columns=columns_to_drop)

# Remove columns containing "_unc" in their name (case-insensitive)
columns_to_drop = [col for col in df.columns if '_unc' in col.lower()]
df = df.drop(columns=columns_to_drop)

#Additional columns to remove
additional_columns_to_remove = ['ï»¿completed_by', 'damage_status', 'ref# (DELETE LATER)',
                                'complete_address', 'building_name_listing','building_name_current', 'notes', 'tornado_name',
                                'tornado_EF', 'tornado_start_lat', 'tornado_start_long','tornado_end_lat', 'tornado_end_long',
                                'national_register_listing_year', 'latitude', 'longitude', 'town','located_in_historic_district',
                                'hazards_present_u']

for col in additional_columns_to_remove:
    if col in df.columns:
        df = df.drop(col, axis=1)
    else:
        print(f"Column '{col}' not found in DataFrame.")



## DUPLICATES BC UPDATED PROVIDED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
columns_removed = []
columns_to_remove = []

for col in df.columns:
    if '_u' in col:
        corresponding_col = col.replace('_u', '_o')
        if corresponding_col in df.columns:
            columns_to_remove.append(corresponding_col)
            columns_removed.append((col, corresponding_col))

filtered_df = df.drop(columns=columns_to_remove)

#print("Removed columns:")
#for col_u, col_o in columns_removed:
#    print(f"- {col_o} (because {col_u} exists)")


# Specify the columns where you want to replace 'un' with ''
columns_to_modify = ['wall_thickness', 'overhang_length_u', 'parapet_height_m'] # Replace with your actual column names

# Iterate through the specified columns and perform the replacement
for column in columns_to_modify:
  if column in df.columns:
    df.loc[df[column] == 'un', column] = ''
    df.loc[df[column] == 'not_applicable', column] = 0
  else:
    print(f"Column '{column}' not found in DataFrame.")

#if column is numeric replace blanks with mean value, if column is object replace blanks with un
for col in df.columns:
  if pd.api.types.is_numeric_dtype(df[col]):
    # Calculate the mean of the numeric column, excluding NaN values
    col_mean = df[col].mean(skipna=True)

    # Replace blanks (empty strings or NaN) with the mean
    df[col] = df[col].replace(r'^\s*$', np.nan, regex=True) #replace blanks with NaN
    df[col] = df[col].fillna(col_mean)
  else:
    # Replace blanks and NaNs with 'un' in string columns
    df[col] = df[col].fillna('un').replace('', 'un', regex=False)

# add random column
df['random_feature'] = np.random.rand(len(df))

# Save the DataFrame to a new CSV file
df.to_csv('cleaned_data_sim.csv', index=False) # index=False prevents writing row indices