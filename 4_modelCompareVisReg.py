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

# Directory to save the combined plot
PLOTS_OUTPUT_DIR = 'benchmarking_plots'
COMBINED_PLOT_FILENAME = 'combined_model_performance_comparison.png'
FULL_PLOT_SAVE_PATH = os.path.join(PLOTS_OUTPUT_DIR, COMBINED_PLOT_FILENAME)
SAVE_PLOT = True # Set to False to only display plot

# Columns from the CSV to include in the subplots
# Ensure these exactly match the headers in your CSV
# Order matters for subplot placement (top-left, top-right, bottom-left, bottom-right)
COLUMNS_FOR_SUBPLOTS = [
    "Mean CV RMSE",
    "Mean CV R2",
    "Mean CV MAE",
    "GridSearch Time (s)" # Or "Mean Fit Time (s)" if using non-tuned results
]

# Determine sort order based on the metric (lower is better for errors, higher for R2)
METRIC_SORT_ORDER = {
    "Mean CV RMSE": True, # Ascending = True (lower is better)
    "Mean CV R2": False, # Ascending = False (higher is better)
    "Mean CV MAE": True, # Ascending = True (lower is better)
    "Mean CV MSE": True, # Ascending = True (lower is better)
    "GridSearch Time (s)": True, # Ascending = True (lower is better)
    "Mean Fit Time (s)": True # Ascending = True (lower is better)
}

# Plotting style
sns.set_style("whitegrid")
# Adjust figsize for a 2x2 layout
PLOT_FIGSIZE = (15, 12) # Width, Height in inches - May need tweaking
PLOT_ROTATION = 60 # Rotation angle for x-axis labels (might need more rotation)
SUPTITLE_FONTSIZE = 18
SUBPLOT_TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 10
TICK_LABEL_FONTSIZE = 9

# --- Helper Functions ---

def load_results_data(csv_path):
    """Loads the benchmarking results CSV."""
    print(f"Loading results data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"  Successfully loaded. Shape: {df.shape}")
        # Attempt to convert potential metric/time columns back to numeric
        numeric_cols = list(METRIC_SORT_ORDER.keys())
        for col in numeric_cols:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isnull().any():
                    print(f"  Warning: Column '{col}' contained non-numeric values, converted to NaN.")
        return df
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_path}")
        exit()
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit()

# --- Main Visualization Script ---

print("Starting Combined Benchmarking Results Visualization...")

# 1. Load Data
results_df = load_results_data(FULL_RESULTS_CSV_PATH)

# 2. Create Output Directory (if saving)
if SAVE_PLOT:
    try:
        os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
        print(f"\nPlot will be saved to: {PLOTS_OUTPUT_DIR}")
    except Exception as e:
        print(f"Warning: Could not create output directory '{PLOTS_OUTPUT_DIR}'. Plot will not be saved. Error: {e}")
        SAVE_PLOT = False

# 3. Create the Combined Plot
print("\nGenerating combined plot...")

if len(COLUMNS_FOR_SUBPLOTS) != 4:
    print(f"Error: This script expects exactly 4 columns in COLUMNS_FOR_SUBPLOTS for a 2x2 grid, but found {len(COLUMNS_FOR_SUBPLOTS)}.")
    exit()

# Create a figure and a 2x2 grid of subplots (axes)
fig, axes = plt.subplots(2, 2, figsize=PLOT_FIGSIZE, sharex=False) # sharex=False allows different sorting on x-axis
# Flatten the 2D array of axes into a 1D array for easy iteration
axes = axes.flatten()

# Loop through the columns and corresponding axes
for i, plot_col in enumerate(COLUMNS_FOR_SUBPLOTS):
    ax = axes[i] # Get the current subplot axis

    print(f"  Processing subplot for: {plot_col}")

    # Check if column exists
    if plot_col not in results_df.columns:
        print(f"    Warning: Column '{plot_col}' not found in DataFrame. Skipping subplot.")
        ax.set_title(f"'{plot_col}'\n(Not Found)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
        ax.set_axis_off() # Turn off the axis if data is missing
        continue

    # Prepare data for the specific subplot
    plot_df = results_df.dropna(subset=[plot_col]).copy()
    if plot_df.empty:
        print(f"    Warning: No valid data found for '{plot_col}' after dropping NaNs. Skipping subplot.")
        ax.set_title(f"'{plot_col}'\n(No Data)", fontsize=SUBPLOT_TITLE_FONTSIZE - 2)
        ax.set_axis_off()
        continue

    # Determine sort order and sort
    ascending = METRIC_SORT_ORDER.get(plot_col, True) # Default to ascending if not in dict
    plot_df = plot_df.sort_values(by=plot_col, ascending=ascending)

    # Create the bar plot on the current axis
    sns.barplot(x='Model', y=plot_col, data=plot_df, palette='viridis', ax=ax)

    # Customize the subplot
    title_suffix = f" ({'Lower' if ascending else 'Higher'} is Better)"
    ax.set_title(f"{plot_col}{title_suffix}", fontsize=SUBPLOT_TITLE_FONTSIZE)
    ax.set_xlabel(None) # Remove individual x-axis labels, models are clear
    ax.set_ylabel(plot_col, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='x', rotation=PLOT_ROTATION, labelsize=TICK_LABEL_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

# Add an overall title to the figure
fig.suptitle("Model Performance Comparison (Mean CV Results)", fontsize=SUPTITLE_FONTSIZE, y=1.03) # Adjust y position if needed

# Adjust layout to prevent labels/titles overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # rect=[left, bottom, right, top] leaves space for suptitle

# 4. Save and Show Plot
if SAVE_PLOT:
    try:
        plt.savefig(FULL_PLOT_SAVE_PATH, dpi=300, bbox_inches='tight')
        print(f"\nCombined plot saved to: {FULL_PLOT_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving combined plot to {FULL_PLOT_SAVE_PATH}: {e}")

print("\nDisplaying combined plot...")
plt.show()
plt.close(fig) # Close the figure after showing/saving

print("\nVisualization script finished.")