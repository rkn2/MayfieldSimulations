import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
INPUT_CSV_PATH = 'feature_importance_sims.csv'
TARGET_COLUMN = 'Difference'
OUTPUT_DIR = 'eda_results'
N_TOP_FEATURES = 15  # How many of the most important features to display

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_important_features(df):
    """
    Identifies the most important features for predicting the target column
    using mutual information.
    """
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found.")
        return

    # Separate target variable y from features X
    y = df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].median())  # Ensure target has no NaNs
    X = df.drop(columns=[TARGET_COLUMN])

    # --- Preprocessing for Mutual Information ---
    X_encoded = X.copy()

    # --- ADDED: Impute NaN values for both numeric and object columns ---
    # Impute numeric columns with the median
    for col in X_encoded.select_dtypes(include=np.number).columns:
        if X_encoded[col].isnull().any():
            median_val = X_encoded[col].median()
            X_encoded[col].fillna(median_val, inplace=True)

    # Label encode object columns (this will handle NaNs by treating them as a separate category)
    for col in X_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        # Convert all values to string to handle mixed types, fill NaNs with 'missing'
        X_encoded[col] = X_encoded[col].astype(str).fillna('missing')
        X_encoded[col] = le.fit_transform(X_encoded[col])

    # --- Calculate Mutual Information ---
    print("Calculating Mutual Information scores... This may take a moment.")
    mi_scores = mutual_info_regression(X_encoded, y, random_state=42)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_encoded.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    # --- Display Results ---
    print(f"\n--- Top {N_TOP_FEATURES} Features Associated with Simulation Error ---")
    print(mi_scores.head(N_TOP_FEATURES))

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    top_features_df = mi_scores.head(N_TOP_FEATURES)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_features_df.values, y=top_features_df.index, palette='viridis')
    plt.title(f'Top {N_TOP_FEATURES} Most Influential Features (Mutual Information)', fontsize=16)
    plt.xlabel('Mutual Information Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'feature_importance_plot.png')
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()


if __name__ == '__main__':
    try:
        data_df = pd.read_csv(INPUT_CSV_PATH)
        find_important_features(data_df)
    except FileNotFoundError:
        print(f"Error: The file was not found at {INPUT_CSV_PATH}")

