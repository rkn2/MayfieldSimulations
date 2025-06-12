import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re  # For sanitizing filenames

# --- Configuration ---
INPUT_CSV_PATH = 'cleaned_data_latlong.csv'
OUTPUT_PLOT_DIR = 'data_visualizations'

# --- USER-DEFINED PLOT VARIABLES ---
# Mandatory:
X_COLUMN = 'longitude'  # Replace with your desired X-axis column name
Y_COLUMN = 'latitude'  # Replace with your desired Y-axis column name

# Optional:
# For coloring points by a categorical or discrete numeric variable.
# Set to None if no color encoding is desired.
HUE_COLUMN = 'foundation_type_u'  # Example: 'foundation_type_u', 'degree_of_damage_u'
# If HUE_COLUMN is numeric with many unique values, consider binning it first
# or ensure it's treated as categorical if appropriate.

# For varying point size by a numeric variable.
# Set to None if no size encoding is desired.
SIZE_COLUMN = None  # Example: 'num__building_area_m2', 'num__roof_slope_u'
# Ensure this column is numeric.

# If using SIZE_COLUMN, you might want to adjust point sizes for better visualization.
# Sizes will be relative to the values in SIZE_COLUMN.
# You can set a min and max size for the points.
POINT_SIZE_MIN = 20
POINT_SIZE_MAX = 200
# sns.scatterplot's `sizes` parameter can take a tuple (min, max)

# --- Plot Customization ---
PLOT_TITLE_PREFIX = "Scatter Plot:"
PLOT_POINT_SIZE_DEFAULT = 60  # Default point size if SIZE_COLUMN is not used
PLOT_ALPHA = 0.75  # Transparency of points


def generate_flexible_scatter_plot():
    """
    Loads data and generates a scatter plot based on user-defined X, Y, Hue (optional), and Size (optional) columns.
    """
    print(f"Loading data from: {INPUT_CSV_PATH}")
    try:
        df = pd.read_csv(INPUT_CSV_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {INPUT_CSV_PATH}. Please ensure the path is correct.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- Verify necessary columns exist ---
    required_cols = [X_COLUMN, Y_COLUMN]
    if HUE_COLUMN:
        required_cols.append(HUE_COLUMN)
    if SIZE_COLUMN:
        required_cols.append(SIZE_COLUMN)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the CSV: {missing_cols}")
        print(f"Available columns are: {df.columns.tolist()}")
        print("Please update the column name variables in the script.")
        return

    # --- Prepare data for plotting ---
    print(f"\nPreparing data for X='{X_COLUMN}', Y='{Y_COLUMN}'...")
    df[X_COLUMN] = pd.to_numeric(df[X_COLUMN], errors='coerce')
    df[Y_COLUMN] = pd.to_numeric(df[Y_COLUMN], errors='coerce')

    plot_data_df = df.copy()  # Work on a copy

    # Handle Hue Column
    if HUE_COLUMN:
        print(f"Using HUE_COLUMN: '{HUE_COLUMN}'")
        plot_data_df.dropna(subset=[HUE_COLUMN], inplace=True)
        # Convert to string to ensure it's treated as categorical by seaborn for distinct colors
        plot_data_df[HUE_COLUMN] = plot_data_df[HUE_COLUMN].astype(str)
        hue_counts = plot_data_df[HUE_COLUMN].value_counts()
        print(f"  Unique values and counts for '{HUE_COLUMN}':\n{hue_counts}")
        if len(hue_counts) > 20:  # Arbitrary limit for too many colors in legend
            print(
                f"  Warning: HUE_COLUMN '{HUE_COLUMN}' has {len(hue_counts)} unique values. Legend might be cluttered.")

    # Handle Size Column
    if SIZE_COLUMN:
        print(f"Using SIZE_COLUMN: '{SIZE_COLUMN}'")
        plot_data_df[SIZE_COLUMN] = pd.to_numeric(plot_data_df[SIZE_COLUMN], errors='coerce')
        plot_data_df.dropna(subset=[SIZE_COLUMN], inplace=True)
        if plot_data_df[SIZE_COLUMN].min() < 0:
            print(
                f"  Warning: SIZE_COLUMN '{SIZE_COLUMN}' contains negative values. Taking absolute values for sizing.")
            plot_data_df[SIZE_COLUMN] = plot_data_df[SIZE_COLUMN].abs()
        print(
            f"  Min/Max for '{SIZE_COLUMN}': {plot_data_df[SIZE_COLUMN].min():.2f} / {plot_data_df[SIZE_COLUMN].max():.2f}")

    # Drop rows where X or Y are NaN after numeric conversion
    original_rows = len(plot_data_df)
    plot_data_df.dropna(subset=[X_COLUMN, Y_COLUMN], inplace=True)
    if len(plot_data_df) < original_rows:
        print(f"  Dropped {original_rows - len(plot_data_df)} rows due to missing/non-numeric X or Y values.")

    if plot_data_df.empty:
        print("No valid data available for plotting after cleaning and preparation.")
        return

    # --- Create the plot ---
    plt.figure(figsize=(14, 10))

    plot_title = f"{PLOT_TITLE_PREFIX} {Y_COLUMN} vs. {X_COLUMN}"
    scatter_kwargs = {
        'data': plot_data_df,
        'x': X_COLUMN,
        'y': Y_COLUMN,
        'alpha': PLOT_ALPHA,
    }

    if HUE_COLUMN:
        scatter_kwargs['hue'] = HUE_COLUMN
        num_unique_hue = plot_data_df[HUE_COLUMN].nunique()
        palette_to_use = 'viridis'  # Default
        if num_unique_hue <= 10:
            palette_to_use = 'tab10'
        elif num_unique_hue <= 20:
            palette_to_use = 'tab20'
        else:
            palette_to_use = 'hls'  # For many categories
        scatter_kwargs['palette'] = palette_to_use
        plot_title += f" (Colored by {HUE_COLUMN})"

    if SIZE_COLUMN:
        scatter_kwargs['size'] = SIZE_COLUMN
        scatter_kwargs['sizes'] = (POINT_SIZE_MIN, POINT_SIZE_MAX)
        plot_title += f" (Sized by {SIZE_COLUMN})"
    else:
        scatter_kwargs['s'] = PLOT_POINT_SIZE_DEFAULT

    print("\nGenerating plot...")
    ax = sns.scatterplot(**scatter_kwargs)

    plt.title(plot_title, fontsize=16)
    plt.xlabel(X_COLUMN, fontsize=14)
    plt.ylabel(Y_COLUMN, fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust legend
    if HUE_COLUMN or SIZE_COLUMN:
        num_legend_items = 0
        if HUE_COLUMN:
            num_legend_items = plot_data_df[HUE_COLUMN].nunique()
        # For size legend, it's usually a few representative sizes

        if num_legend_items > 8 or SIZE_COLUMN:  # Move legend out if many hue categories or if size is used
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title_fontsize='13', fontsize='10')
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            plt.legend(title_fontsize='13', fontsize='11')
            plt.tight_layout()
    else:
        plt.tight_layout()

    # --- Save the plot ---
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

    # Create a safe filename
    filename_parts = [Y_COLUMN, "vs", X_COLUMN]
    if HUE_COLUMN: filename_parts.extend(["by", HUE_COLUMN])
    if SIZE_COLUMN: filename_parts.extend(["sized_by", SIZE_COLUMN])
    safe_filename_base = re.sub(r'\W+', '_', "_".join(filename_parts).lower())
    output_filename = f"{safe_filename_base}_plot.png"
    output_path = os.path.join(OUTPUT_PLOT_DIR, output_filename)

    try:
        plt.savefig(output_path)
        print(f"\nPlot saved successfully to: {output_path}")
    except Exception as e:
        print(f"\nError saving plot: {e}")

    plt.show()


if __name__ == '__main__':
    generate_flexible_scatter_plot()
