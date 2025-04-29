import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# --- Configuration ---
RESULTS_DIR = 'processed_ml_data' # Directory where results CSV is saved
RESULTS_FILENAME = 'model_tuned_cv_benchmarking_results.csv'
FULL_RESULTS_CSV_PATH = os.path.join(RESULTS_DIR, RESULTS_FILENAME)

# Directory to save the plots
PLOTS_OUTPUT_DIR = 'benchmarking_plots'
SAVE_PLOTS = True # Set to False to only display plots

# Columns from the CSV to visualize
METRIC_COLUMNS_TO_PLOT = [
    "Mean CV RMSE",
    "Mean CV R2",
    "Mean CV MAE",
    # Add "Mean CV MSE" if desired, but RMSE is often preferred
]
TIME_COLUMN_TO_PLOT = "GridSearch Time (s)" # Or "Mean Fit Time (s)" if using the non-tuned results

# Determine sort order based on the metric (lower is better for errors, higher for R2)
# This dictionary helps automate plot sorting and titles
METRIC_SORT_ORDER = {
    "Mean CV RMSE": True, # Ascending = True (lower is better)
    "Mean CV R2": False, # Ascending = False (higher is better)
    "Mean CV MAE": True, # Ascending = True (lower is better)
    "Mean CV MSE": True, # Ascending = True (lower is better)
    "GridSearch Time (s)": True # Ascending = True (lower is better)
}

# Plotting style
sns.set_style("whitegrid")
PLOT_FIGSIZE = (12, 7) # Width, Height in inches
PLOT_ROTATION = 45 # Rotation angle for x-axis labels

# --- Helper Functions ---

def load_results_data(csv_path):
    """Loads the benchmarking results CSV."""
    print(f"Loading results data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"  Successfully loaded. Shape: {df.shape}")
        # Attempt to convert potential metric/time columns back to numeric if read as object
        numeric_cols = list(METRIC_SORT_ORDER.keys()) # Get all columns we might plot
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    print(f"  Warning: Column '{col}' contained non-numeric values after loading, converted to NaN.")
        return df
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_path}")
        print("Please ensure the benchmarking script ran successfully and saved the results.")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

def plot_comparison(df, plot_col, ascending_sort, title, save_path=None, figsize=PLOT_FIGSIZE, rotation=PLOT_ROTATION):
    """Creates a sorted bar plot comparing models for a given column."""
    if plot_col not in df.columns:
        print(f"Warning: Column '{plot_col}' not found in DataFrame. Skipping plot.")
        return

    # Drop rows where the plotting column is NaN for cleaner plotting
    plot_df = df.dropna(subset=[plot_col]).copy()
    if plot_df.empty:
        print(f"Warning: No valid data found for column '{plot_col}' after dropping NaNs. Skipping plot.")
        return

    # Sort data for better visualization
    plot_df = plot_df.sort_values(by=plot_col, ascending=ascending_sort)

    plt.figure(figsize=figsize)
    barplot = sns.barplot(x='Model', y=plot_col, data=plot_df, palette='viridis')

    # Add value labels on top of bars (optional, can get crowded)
    # for index, row in plot_df.iterrows():
    #    barplot.text(index, row[plot_col], f"{row[plot_col]:.3f}", color='black', ha="center")

    plt.title(title, fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel(plot_col, fontsize=12)
    plt.xticks(rotation=rotation, ha='right') # Rotate labels for readability
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")

    plt.show()
    plt.close() # Close the figure after showing/saving

# --- Main Visualization Script ---

print("Starting Benchmarking Results Visualization...")

# 1. Load Data
results_df = load_results_data(FULL_RESULTS_CSV_PATH)

# 2. Create Output Directory (if saving)
if SAVE_PLOTS:
    try:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"\nPlots will be saved to: {PLOTS_OUTPUT_DIR}")
    except Exception as e:
        print(f"Warning: Could not create output directory '{PLOTS_OUTPUT_DIR}'. Plots will not be saved. Error: {e}")
        SAVE_PLOTS = False # Disable saving if directory creation fails

# 3. Generate and Save/Show Plots
print("\nGenerating plots...")

# Plot comparisons for each specified metric
for metric in METRIC_COLUMNS_TO_PLOT:
    if metric in results_df.columns:
        ascending = METRIC_SORT_ORDER.get(metric, True) # Default to ascending sort if not specified
        title = f"Model Comparison by {metric} ({'Lower' if ascending else 'Higher'} is Better)"
        save_filename = f"comparison_{metric.lower().replace(' ', '_')}.png" if SAVE_PLOTS else None
        full_save_path = os.path.join(PLOTS_OUTPUT_DIR, save_filename) if save_filename else None

        print(f"\nPlotting: {title}")
        plot_comparison(
            df=results_df,
            plot_col=metric,
            ascending_sort=ascending,
            title=title,
            save_path=full_save_path
        )
    else:
        print(f"\nSkipping plot: Metric column '{metric}' not found in results.")

# Plot comparison for tuning/fit time
if TIME_COLUMN_TO_PLOT in results_df.columns:
    ascending = METRIC_SORT_ORDER.get(TIME_COLUMN_TO_PLOT, True)
    title = f"Model Comparison by {TIME_COLUMN_TO_PLOT} (Lower is Better)"
    save_filename = f"comparison_{TIME_COLUMN_TO_PLOT.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png" if SAVE_PLOTS else None
    full_save_path = os.path.join(PLOTS_OUTPUT_DIR, save_filename) if save_filename else None

    print(f"\nPlotting: {title}")
    plot_comparison(
        df=results_df,
        plot_col=TIME_COLUMN_TO_PLOT,
        ascending_sort=ascending,
        title=title,
        save_path=full_save_path
    )
else:
    print(f"\nSkipping plot: Time column '{TIME_COLUMN_TO_PLOT}' not found in results.")


print("\nVisualization script finished.")