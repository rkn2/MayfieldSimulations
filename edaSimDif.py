import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


# --- Configuration ---
# Update this path to your data file
INPUT_CSV_PATH = 'feature_importance_sims.csv'
TARGET_COLUMN = 'Difference'
OUTPUT_DIR = 'eda_results'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_difference(df):
    """
    Analyzes the distribution of the target variable and prints key statistics.
    """
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the CSV file.")
        return

    # --- Calculate Statistics ---
    difference_series = df[TARGET_COLUMN]
    mean_diff = difference_series.mean()
    median_diff = difference_series.median()
    std_diff = difference_series.std()
    skewness = difference_series.skew()

    print("--- Analysis of Simulation 'Difference' ---")
    print(f"Mean Difference: {mean_diff:.2f}")
    print(f"Median Difference: {median_diff:.2f}")
    print(f"Standard Deviation: {std_diff:.2f}")
    print(f"Skewness: {skewness:.2f}")

    # --- Interpretation ---
    if mean_diff > 0.1:
        print("\nInterpretation: The simulation tends to UNDERESTIMATE the damage level.")
    elif mean_diff < -0.1:
        print("\nInterpretation: The simulation tends to OVERESTIMATE the damage level.")
    else:
        print("\nInterpretation: The simulation appears to be relatively unbiased on average.")

    if skewness > 0.5:
        print("The distribution is positively skewed, suggesting more cases of underestimation.")
    elif skewness < -0.5:
        print("The distribution is negatively skewed, suggesting more cases of overestimation.")

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    sns.histplot(difference_series, kde=True, ax=axes[0], bins=10)
    axes[0].set_title(f'Distribution of Simulation Difference', fontsize=16)
    axes[0].set_xlabel('Difference (Actual - Simulated)', fontsize=12)
    axes[0].set_ylabel('Number of Buildings', fontsize=12)
    axes[0].axvline(mean_diff, color='red', linestyle='--', label=f'Mean: {mean_diff:.2f}')
    axes[0].axvline(median_diff, color='green', linestyle='-', label=f'Median: {median_diff:.2f}')
    axes[0].legend()

    # Boxplot
    sns.boxplot(y=difference_series, ax=axes[1])
    axes[1].set_title('Boxplot of Simulation Difference', fontsize=16)
    axes[1].set_ylabel('Difference (Actual - Simulated)', fontsize=12)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'difference_analysis_plots.png')
    plt.savefig(plot_path)
    print(f"\nPlots saved to: {plot_path}")
    plt.show()


if __name__ == '__main__':
    try:
        data_df = pd.read_csv(INPUT_CSV_PATH)
        analyze_difference(data_df)
    except FileNotFoundError:
        print(f"Error: The file was not found at {INPUT_CSV_PATH}")

