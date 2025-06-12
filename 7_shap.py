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

    # 3. Initialize SHAP Explainer and Calculate SHAP values for ALL classes
    print("\n--- Calculating SHAP Values ---")
    try:
        if "XGBoost" in str(type(model)) or "LGBM" in str(type(model)) or "RandomForest" in str(
                type(model)) or "DecisionTree" in str(type(model)):
            print("Tree-based model detected. Using shap.TreeExplainer for efficiency.")
            explainer = shap.TreeExplainer(model, background_data)
            # Use the modern explainer call which returns a single Explanation object
            shap_explanation_object = explainer(X_test_sample)
        else:
            print("Non-tree model detected. Using shap.KernelExplainer. This might take some time...")
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            shap_values_raw = explainer.shap_values(X_test_sample)
            # Manually create the explanation object for KernelExplainer
            shap_explanation_object = shap.Explanation(
                values=shap_values_raw,
                base_values=[explainer.expected_value[i] for i in range(len(shap_values_raw))],
                data=X_test_sample.values,
                feature_names=X_test_sample.columns.tolist()
            )

        print("SHAP values calculated successfully.")

    except Exception as e:
        print(f"Error during SHAP value calculation: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure the 'shap' library is installed (`pip install shap`).")
        exit()

    unique_classes = sorted(np.unique(y_test))
    class_names = [f"Class {c}" for c in unique_classes]
    print(f"\nFound unique classes in test set: {unique_classes}. Analyzing all classes.")

    # 4. Create and Save Composite Bar Chart
    print("\n--- Generating Composite SHAP Bar Plot for All Classes ---")

    # Calculate mean absolute SHAP values for each class directly from the explanation object
    mean_abs_shap_per_class = [np.abs(shap_explanation_object.values[:, :, i]).mean(0) for i in
                               range(len(unique_classes))]

    feature_importance_df = pd.DataFrame(
        data=np.array(mean_abs_shap_per_class).T,
        index=X_test_sample.columns,
        columns=class_names
    )

    feature_importance_df['Total Importance'] = feature_importance_df.sum(axis=1)
    top_features = feature_importance_df.nlargest(N_TOP_FEATURES_TO_PLOT, 'Total Importance')

    top_features.drop(columns=['Total Importance']).plot(
        kind='bar', figsize=(16, 9), width=0.8, stacked=False, colormap='viridis'
    )

    plt.title(f'Top {N_TOP_FEATURES_TO_PLOT} Feature Importances by Class', fontsize=16)
    plt.ylabel('Mean Absolute SHAP Value (Impact on prediction)', fontsize=12)
    plt.xlabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary_bar_composite.png'))
    plt.close()
    print("    Saved composite bar chart to shap_summary_bar_composite.png")

    # 5. Loop to generate individual Beeswarm plots and console output
    for i, target_class_to_explain in enumerate(unique_classes):
        print(f"\n============================================================")
        print(f"  ANALYZING SHAP VALUES FOR CLASS: {target_class_to_explain}")
        print(f"============================================================")

        # Print console output
        class_importance_df = pd.DataFrame({
            'Feature': X_test_sample.columns,
            'Mean Absolute SHAP Value': feature_importance_df[f'Class {target_class_to_explain}']
        }).sort_values(by='Mean Absolute SHAP Value', ascending=False)

        print(f"Top {N_TOP_FEATURES_TO_PLOT} features influencing predictions for class {target_class_to_explain}:")
        print(class_importance_df.head(N_TOP_FEATURES_TO_PLOT).to_string(index=False))

        # Generate and Save Beeswarm plot
        print(f"\n  Generating SHAP Beeswarm Summary Plot for Class {target_class_to_explain}...")
        plt.figure()
        # *** FIX: Use the more robust shap.plots.beeswarm directly with the sliced Explanation object ***
        shap.plots.beeswarm(shap_explanation_object[:, :, i], max_display=N_TOP_FEATURES_TO_PLOT, show=False)
        plt.title(f'SHAP Summary Plot (for class {target_class_to_explain})', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'shap_summary_beeswarm_class_{target_class_to_explain}.png'))
        plt.close()
        print(f"    Saved shap_summary_beeswarm_class_{target_class_to_explain}.png")

    print("\nSHAP analysis complete. All plots saved in the 'shap_plots' directory.")


if __name__ == "__main__":
    main()

