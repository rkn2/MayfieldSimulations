import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.pipeline import Pipeline
import warnings

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
MODEL_SAVE_PATH = os.path.join(DATA_DIR, 'best_tuned_classifier.pkl')
PREPROCESSOR_SAVE_PATH = os.path.join(DATA_DIR, 'preprocessor.pkl')
RESULTS_DIR = 'pdp_results'

# --- Define the features you want to plot ---
FEATURES_TO_PLOT = [
    'buidling_height_m',
    'longitude',
    'year_built_u',
    'building_area_m2'
]

# Define the labels for your target classes
CLASS_NAMES = ['Undamaged', 'Damaged', 'Destroyed']
# The actual class values used by the model
CLASS_INDICES = [0, 1, 2]


# --- Helper Functions ---
def load_tool(file_path, description="tool"):
    """Loads a saved object."""
    print(f"Loading {description} from {file_path}...")
    try:
        tool = joblib.load(file_path)
        print(f"  Successfully loaded: {description}")
        return tool
    except FileNotFoundError:
        print(f"Error: {description} file not found at {file_path}. Exiting.")
        exit()
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}. Exiting.")
        exit()


# --- Main Script ---
def main():
    print("--- Starting Partial Dependence Plot (PDP) Generation ---")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # 1. Load tools
    model = load_tool(MODEL_SAVE_PATH, "Best Model")
    preprocessor = load_tool(PREPROCESSOR_SAVE_PATH, "Preprocessor")

    # 2. Create pipeline
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    print("\nSuccessfully created full model pipeline.")

    # 3. Load original data
    try:
        X_test_unprocessed = joblib.load(os.path.join(DATA_DIR, 'X_test_processed.pkl'))

        numeric_features_original = preprocessor.transformers_[0][2]
        categorical_features_original = preprocessor.transformers_[1][2]
        all_original_cols = numeric_features_original + categorical_features_original
        X_test_df = pd.DataFrame(X_test_unprocessed, columns=all_original_cols)

        print("Ensuring correct data types...")
        for col in categorical_features_original:
            if col in X_test_df.columns:
                X_test_df[col] = X_test_df[col].astype(str)

        for col in numeric_features_original:
            if col in X_test_df.columns:
                X_test_df[col] = pd.to_numeric(X_test_df[col], errors='coerce')
                # ### MODIFIED: Handle cases where a column becomes all NaNs ###
                if X_test_df[col].isnull().all():
                    X_test_df[col] = X_test_df[col].fillna(0)
                elif X_test_df[col].isnull().any():
                    median_val = X_test_df[col].median()
                    X_test_df[col] = X_test_df[col].fillna(median_val)

        print("Successfully loaded and prepared original test data for PDP background.")
    except Exception as e:
        print(f"Could not load original test data. Error: {e}")
        return

    # 4. Generate PDP plots
    print(f"\nGenerating PDP plots for features: {FEATURES_TO_PLOT}...")

    colors = sns.color_palette('viridis', n_colors=len(CLASS_NAMES))

    for feature in FEATURES_TO_PLOT:
        if feature not in X_test_df.columns:
            print(f"  - WARNING: Feature '{feature}' not found. Skipping.")
            continue

        print(f"  - Plotting for feature: '{feature}'...")
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            ### MODIFIED: Unpack the tuple returned by partial_dependence ###
            y_values_all_classes, x_values_list = partial_dependence(
                full_pipeline,
                X_test_df,
                features=[feature],
                grid_resolution=50
            )
            x_values = x_values_list[0]

            # Loop through the results for each class and plot them
            for i, y_values in enumerate(y_values_all_classes):
                ax.plot(x_values, y_values, color=colors[i], linewidth=3, alpha=0.8, label=CLASS_NAMES[i])

            ax.set_title(f'Partial Dependence Plot for\n"{feature}"', fontsize=16)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel("Partial Dependence (Probability)", fontsize=12)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.legend(title="Damage Class")

            # Save the plot
            plot_filename = f"pdp_{feature}.png"
            plot_save_path = os.path.join(RESULTS_DIR, plot_filename)
            plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
            print(f"    Saved plot to {plot_save_path}")
            plt.close(fig)

        except Exception as e:
            print(f"    - ERROR: Could not generate PDP for '{feature}'. Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- PDP Generation Complete ---")


if __name__ == "__main__":
    main()
