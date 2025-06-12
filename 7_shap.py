import pandas as pd
import numpy as np
import os
import joblib
import warnings
import matplotlib.pyplot as plt
import shap

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
MODEL_DIR = 'processed_ml_data'  # Assuming the best model is saved here
RESULTS_DIR = 'shap_plots'  # Directory to save SHAP plots
BEST_MODEL_FILENAME = 'best_tuned_classifier.pkl'
PREPROCESSOR_FILENAME = 'preprocessor.pkl'

# --- Paths ---
BEST_MODEL_PATH = os.path.join(MODEL_DIR, BEST_MODEL_FILENAME)
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, PREPROCESSOR_FILENAME)
X_TRAIN_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
X_TEST_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
Y_TEST_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

# --- SHAP Configuration ---
# Using a subset of the test data for SHAP value calculation can speed up the process significantly.
# Set to None to use the whole test set (can be slow).
N_SHAP_SAMPLES = 1000
N_BACKGROUND_SAMPLES = 200  # Number of samples from training data for the background dataset
N_TOP_FEATURES_TO_PLOT = 15  # Number of features for dependence plots


def load_data_and_model():
    """Loads all necessary data and the trained model."""
    print("--- Loading Data and Model ---")
    if not all(os.path.exists(p) for p in [X_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH, BEST_MODEL_PATH, PREPROCESSOR_PATH]):
        print("Error: Not all required files were found. Please ensure scripts 1, 2, and 3 have been run successfully.")
        exit()

    try:
        X_train = joblib.load(X_TRAIN_PATH)
        X_test = joblib.load(X_TEST_PATH)
        y_test = joblib.load(Y_TEST_PATH)
        model = joblib.load(BEST_MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("  Successfully loaded all required files.")

        # Ensure data is in DataFrame format for feature names
        if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
            print(
                "Warning: Processed data is not in a DataFrame format. Attempting to get feature names from preprocessor.")
            try:
                feature_names = preprocessor.get_feature_names_out()
                X_train = pd.DataFrame(X_train, columns=feature_names)
                X_test = pd.DataFrame(X_test, columns=feature_names)
            except Exception as e:
                print(
                    f"Error: Could not construct DataFrame from processed data. Feature names will be missing. Error: {e}")
                exit()

        return X_train, X_test, y_test, model, preprocessor
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        exit()


def main():
    """Main function to run SHAP analysis."""
    warnings.filterwarnings("ignore", category=UserWarning)

    # 1. Load data
    X_train, X_test, y_test, model, preprocessor = load_data_and_model()

    print(f"\nModel loaded: {type(model).__name__}")

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"SHAP plots will be saved to '{RESULTS_DIR}/'")

    # 2. Prepare data for SHAP
    if N_SHAP_SAMPLES and N_SHAP_SAMPLES < len(X_test):
        print(f"\nUsing a random sample of {N_SHAP_SAMPLES} data points from the test set for SHAP analysis.")
        X_test_sample = X_test.sample(N_SHAP_SAMPLES, random_state=42)
    else:
        print("\nUsing the full test set for SHAP analysis.")
        X_test_sample = X_test

    n_background = min(N_BACKGROUND_SAMPLES, len(X_train))
    print(f"Creating background data with {n_background} samples from the training set...")
    if n_background == 0:
        print("Error: Training data is empty. Cannot create a background dataset for SHAP.")
        exit()
    background_data = X_train.sample(n_background, random_state=42)

    # 3. Initialize SHAP Explainer and Calculate SHAP values
    print("\n--- Calculating SHAP Values ---")
    try:
        if "XGBoost" in str(type(model)) or "LGBM" in str(type(model)) or "RandomForest" in str(
                type(model)) or "DecisionTree" in str(type(model)):
            print("Tree-based model detected. Using shap.TreeExplainer for efficiency.")
            explainer = shap.TreeExplainer(model, background_data)
            shap_values_raw = explainer.shap_values(X_test_sample)
        else:
            print("Non-tree model detected. Using shap.KernelExplainer. This might take some time...")
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            shap_values_raw = explainer.shap_values(X_test_sample)

        print("SHAP values calculated successfully.")

    except Exception as e:
        print(f"Error during SHAP value calculation: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure the 'shap' library is installed (`pip install shap`).")
        exit()

    target_class_to_explain = pd.Series(y_test).mode()[0]
    print(f"\nAnalyzing SHAP values for the most common target class: {target_class_to_explain}")

    if isinstance(shap_values_raw, list):
        shap_values_for_class = shap_values_raw[target_class_to_explain]
    else:
        shap_values_for_class = shap_values_raw

    # 4. Calculate and Print SHAP Feature Importance Results
    print("\n--- SHAP Feature Importance Results (Console Output) ---")

    mean_abs_shap = np.abs(shap_values_for_class).mean(0)

    # *** FIX: Explicitly convert data to simple lists before creating the DataFrame ***
    feature_names_list = X_test_sample.columns.tolist()
    shap_values_list = mean_abs_shap.tolist()

    if len(feature_names_list) != len(shap_values_list):
        print("Error: Mismatch between number of features and number of SHAP values. Cannot create results table.")
    else:
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names_list,
            'Mean Absolute SHAP Value': shap_values_list
        })

        feature_importance_df.sort_values(by='Mean Absolute SHAP Value', ascending=False, inplace=True)

        print(f"Top {N_TOP_FEATURES_TO_PLOT} features influencing predictions for class {target_class_to_explain}:")
        print(feature_importance_df.head(N_TOP_FEATURES_TO_PLOT).to_string(index=False))

    # 5. Generate and Save SHAP Plots
    print("\n--- Generating SHAP Plots ---")

    # Beeswarm plot (Global Feature Importance)
    print("  Generating SHAP Beeswarm Summary Plot...")
    plt.figure()
    shap.summary_plot(shap_values_for_class, X_test_sample, plot_type="beeswarm", max_display=N_TOP_FEATURES_TO_PLOT,
                      show=False)
    plt.title(f'SHAP Summary Plot (for class {target_class_to_explain})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary_beeswarm.png'))
    plt.close()
    print("    Saved shap_summary_beeswarm.png")

    # Bar plot (Global Feature Importance)
    print("  Generating SHAP Bar Plot...")
    plt.figure()
    shap.summary_plot(shap_values_for_class, X_test_sample, plot_type="bar", max_display=N_TOP_FEATURES_TO_PLOT,
                      show=False)
    plt.title(f'SHAP Mean Absolute Feature Importance (for class {target_class_to_explain})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary_bar.png'))
    plt.close()
    print("    Saved shap_summary_bar.png")

    print("\nSHAP analysis complete. Summary plots are saved in the 'shap_plots' directory.")


if __name__ == "__main__":
    main()
