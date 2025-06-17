import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats # Import for statistical tests


# --- Configuration ---
# Update this path to your data file
INPUT_CSV_PATH = 'feature_importance_sims.csv'
TARGET_COLUMN = 'Difference' # This is 'Actual - Simulated'
# Add columns for statistical significance test
SIMULATED_DAMAGE_COLUMN = 'simulated_damage'
ESTIMATED_DAMAGE_COLUMN = 'estimated'
OUTPUT_DIR = 'eda_results'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_difference(df):
    """
    Analyzes the distribution of the target variable ('Difference') and prints key statistics.
    Also checks for statistical significance between 'simulated_damage' and 'estimated' values.
    """
    # --- Check for target column ---
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the CSV file.")
        return

    # --- Calculate Statistics for 'Difference' ---
    difference_series = df[TARGET_COLUMN]
    mean_diff = difference_series.mean()
    median_diff = difference_series.median()
    std_diff = difference_series.std()
    skewness = difference_series.skew()

    print("--- Analysis of Simulation 'Difference' (Actual - Simulated) ---")
    print(f"Mean Difference: {mean_diff:.2f}")
    print(f"Median Difference: {median_diff:.2f}")
    print(f"Standard Deviation: {std_diff:.2f}")
    print(f"Skewness: {skewness:.2f}")

    # --- Interpretation of 'Difference' ---
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

    # --- Plotting 'Difference' ---
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
    plt.close() # Close the plot to free memory


    # --- Statistical Significance Test: Simulated vs. Estimated ---
    print("\n--- Statistical Significance Test: Simulated vs. Estimated Damage ---")

    # Check if required columns exist
    if SIMULATED_DAMAGE_COLUMN not in df.columns or ESTIMATED_DAMAGE_COLUMN not in df.columns:
        print(f"Error: Missing '{SIMULATED_DAMAGE_COLUMN}' or '{ESTIMATED_DAMAGE_COLUMN}' column for statistical test.")
        return

    # Prepare data for paired t-test
    # Drop rows where either simulated or estimated damage is NaN
    paired_data = df[[SIMULATED_DAMAGE_COLUMN, ESTIMATED_DAMAGE_COLUMN]].dropna()

    if paired_data.empty:
        print("No complete pairs of 'simulated_damage' and 'estimated' found for the statistical test.")
        return

    simulated_values = pd.to_numeric(paired_data[SIMULATED_DAMAGE_COLUMN], errors='coerce')
    estimated_values = pd.to_numeric(paired_data[ESTIMATED_DAMAGE_COLUMN], errors='coerce')

    # Drop any rows that became NaN after numeric conversion
    valid_pairs = pd.DataFrame({'simulated': simulated_values, 'estimated': estimated_values}).dropna()

    if valid_pairs.empty:
        print("No valid numeric pairs of 'simulated_damage' and 'estimated' after conversion for the statistical test.")
        return

    # Perform a paired t-test (also known as dependent t-test)
    # This tests if the mean difference between two related sets of observations is zero.
    t_statistic, p_value = stats.ttest_rel(valid_pairs['simulated'], valid_pairs['estimated'])

    alpha = 0.05 # Significance level

    print(f"Paired t-test results (Simulated vs. Estimated):")
    print(f"  T-statistic: {t_statistic:.3f}")
    print(f"  P-value: {p_value:.3f}")

    if p_value < alpha:
        print(f"\nInterpretation: With a p-value of {p_value:.3f} (which is less than {alpha}), we REJECT the null hypothesis.")
        print("This suggests there is a STATISTICALLY SIGNIFICANT difference between the mean simulated and mean estimated damage levels.")
        if t_statistic > 0:
            print(f"On average, simulated damage is higher than estimated damage.")
        else:
            print(f"On average, estimated damage is higher than simulated damage.")
    else:
        print(f"\nInterpretation: With a p-value of {p_value:.3f} (which is greater than or equal to {alpha}), we FAIL TO REJECT the null hypothesis.")
        print("This suggests there is NO STATISTICALLY SIGNIFICANT difference between the mean simulated and mean estimated damage levels.")


if __name__ == '__main__':
    try:
        data_df = pd.read_csv(INPUT_CSV_PATH)
        analyze_difference(data_df)
    except FileNotFoundError:
        print(f"Error: The file was not found at {INPUT_CSV_PATH}")

