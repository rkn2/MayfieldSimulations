
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder
import numpy as np
import csv
from datetime import datetime
plt.interactive(True)


df = pd.read_csv('../QuadState_Tornado_DataInputv2.csv', delimiter=',', encoding='latin-1')
#
## REMOVING COLUMNS

#drop any row where degree_of_damage_u is blank
df = df.dropna(subset=['degree_of_damage_u'])
df = df.reset_index(drop=True)

## VARIATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Calculate the variation for each column (you can use other methods like standard deviation)
variations = df.nunique()

# Set a threshold for variation (e.g., columns with only one unique value are removed)
threshold = 1

# Filter out columns with variation below the threshold
columns_to_keep = variations[variations > threshold].index
#print(len(df))
#print(len(columns_to_keep))
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
additional_columns_to_remove = ['completed_by', 'damage_status', 'ref# (DELETE LATER)',
                                'complete_address', 'building_name_listing','building_name_current', 'notes', 'tornado_name',
                                'tornado_EF', 'tornado_start_lat', 'tornado_start_long','tornado_end_lat', 'tornado_end_long',
                                'national_register_listing_year', 'town','located_in_historic_district',
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
df.to_csv('cleaned_data_latlong.csv', index=False) # index=False prevents writing row indices

# Generate a timestamp for the filename
#timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "../feature_importance_results_latlong.csv"


# Load the cleaned dataset
df = pd.read_csv('../cleaned_data_latlong.csv')

## SET UP X AND Y
# what is my initial Y
# degree_of_damage_u
y = df['degree_of_damage_u']

# what is my X
# Find columns containing "damage" (case-insensitive)
damage_columns = [col for col in df.columns if 'damage' in col.lower()]
# Drop the identified columns
df = df.drop(columns=damage_columns)

# Find columns containing "exist" (case-insensitive)
exist_columns = [col for col in df.columns if 'status_u' in col.lower() or 'exist' in col.lower() or 'demolish' in col.lower() or 'failure' in col.lower() or 'after' in col.lower()]
# Drop the identified columns
df = df.drop(columns=exist_columns)

# Save the modified DataFrame to a new CSV file
df.to_csv('cleaned_data_no_damage_latlong.csv', index=False)

# now load this in as X
X = pd.read_csv('../cleaned_data_no_damage_latlong.csv')

# CORR and Mutual info
# Encode categorical variables
le = LabelEncoder()
for col in X.select_dtypes(include=['object']):
    X[col] = le.fit_transform(X[col])

# Correlation Analysis
correlations = X.corrwith(y)
top_correlations = correlations.abs().sort_values(ascending=False).head(10)

# Multiple runs of Mutual Information
n_runs = 100
mi_scores_multiple = []

for _ in range(n_runs):
    mi_scores = mutual_info_regression(X, y)
    mi_scores_multiple.append(mi_scores)

mi_scores_df = pd.DataFrame(mi_scores_multiple, columns=X.columns)

# Get top 10 features based on median MI score
top_mi_features = mi_scores_df.median().sort_values(ascending=False).head(10).index

# Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))

# Correlation plot
sns.barplot(x=top_correlations.index, y=top_correlations.values, ax=ax1)
ax1.set_title('Top 10 Correlated Features')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set_ylabel('Absolute Correlation')

# Mutual Information box plot
sns.boxplot(data=mi_scores_df[top_mi_features], ax=ax2)
ax2.set_title(f'Top 10 Features by Mutual Information (Over {n_runs} Runs)')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.set_ylabel('Mutual Information Score')

plt.tight_layout()



# Save the figure
figure_filename = '../feature_importance_results_latlong.png'
plt.savefig(figure_filename, dpi=300, bbox_inches='tight')

print(f"Figure has been saved to {figure_filename}")

# Clear the current figure
plt.clf()

# save info

# Create a list to store all results
results = []

# Add top correlated features
results.append(["Top Correlated Features", ""])
for feature, correlation in top_correlations.items():
    results.append([feature, correlation])

# Add a blank row for separation
results.append(["", ""])

# Add top features by Median Mutual Information
results.append(["Top Features by Median Mutual Information", ""])
mi_median = mi_scores_df[top_mi_features].median().sort_values(ascending=False)
for feature, score in mi_median.items():
    results.append([feature, score])

# Write results to CSV file
with open(filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Feature", "Score"])  # Header
    csvwriter.writerows(results)

print(f"Results have been saved to {filename}")