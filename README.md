# Tornado Damage Data Analysis Pipeline

This repository implements a data analysis pipeline to process, analyze, and report on simulated tornado damage data. Each script serves a specific purpose, handling data from initial cleaning to advanced model interpretability and automated report generation.

## Setup and Installation

To set up the project and install the necessary dependencies, follow these steps:

1.  **Clone the Repository (if not already done):**

    ```
    git clone <repository_url>
    cd MayfieldSimulations
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Install all required libraries using the `requirements.txt` file:

    ```
    pip install -r requirements.txt
    ```

## Pipeline Execution

The entire data analysis pipeline is orchestrated by the `main.py` script. To run the complete pipeline, execute the following command from the root directory of the repository:

The pipeline will execute the scripts sequentially, logging its progress to the console and a file specified in `config.py`.

## Configuration

The `config.py` file centralizes all configurable parameters for the pipeline, including file paths, thresholds, model settings, and visualization preferences. Users can modify this file to adjust the behavior of the pipeline without altering the core logic of the scripts.

## Pipeline Details

The pipeline consists of the following scripts, executed sequentially by `main.py`:

### `1_dataCleaning.py`

This script focuses on cleaning the raw input data, transforming it into a more usable format for subsequent machine learning tasks.

* **Input Data**: `feature_importance_sims.csv` (path configured in `config.INPUT_CSV_PATH`).

* **Key Operations**:

    * **Initial Row Filtering**: Drops rows where the target variable (`Difference`) is missing.

    * **Low Variation Column Removal**: Removes columns with very few unique values (less than `LOW_VARIATION_THRESHOLD`, set to 1 in `config.py`) to eliminate non-informative features.

    * **Column Dropping**: Removes columns based on specified keywords (e.g., 'photos', 'details', 'prop\_', '\_unc') and a predefined list of specific column names (e.g., 'completed_by', 'damage_status', 'notes').

    * **Older Version Column Removal**: If both an original (`_o`) and an updated (`_u`) version of a column exist, the older (`_o`) version is dropped.

    * **Specific Value Replacements**: Replaces non-numeric placeholder values (e.g., 'un', 'not_applicable') with numerical equivalents (e.g., 0) in specific columns, followed by conversion to numeric type.

    * **Missing Value Imputation**: Numeric columns are imputed with their mean, while object/string columns have missing or empty values replaced with 'un'.

    * **Random Feature Addition**: Adds a 'random_feature' for modeling robustness checks.

* **Output Data**: `cleaned_data.csv` (saved to `config.CLEANED_CSV_PATH`).

### `2_dataPreprocessing.py`

This script prepares the cleaned data for machine learning by handling the target variable, applying feature transformations, and optionally performing feature selection and data balancing.

* **Input Data**: `cleaned_data.csv` (output from `1_dataCleaning.py`).

* **Key Operations**:

    * **Target Variable Separation**: Extracts the `Difference` column as the target variable (`y`).

    * **Class Reduction Strategy**: Though present, `REDUCE_CLASSES_STRATEGY` is set to `None` in `config.py`, indicating this step is bypassed for the current regression task.

    * **Feature Filtering**: Removes features from the input data (`X`) that contain keywords specified in `config.KEYWORDS_TO_REMOVE_FROM_X` (e.g., 'damage', 'status_u', 'exist', 'demolish') to avoid data leakage or redundant information.

    * **Data Splitting**: Divides the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using a `TEST_SIZE` of 0.2 and a fixed `RANDOM_STATE`.

    * **Feature Preprocessing**: Numeric features are standardized using `StandardScaler`, and categorical features are converted to numerical format using `OneHotEncoder`.

    * **Recursive Feature Elimination (RFE)**: If `PERFORM_RFE` is `True` in `config.py`, RFE is performed to select the top `N_FEATURES_TO_SELECT` features using a `RandomForestClassifier`.

    * **Data Balancing**: Though present, `BALANCING_METHOD` is set to `None` in `config.py`, indicating this step is bypassed for the current regression task.

* **Output Data**: `X_train_processed.pkl`, `y_train.pkl`, `X_test_processed.pkl`, `y_test.pkl`, and `preprocessor.pkl` (for transforming new data) saved in the `processed_ml_data` directory. Also generates `data_distribution_summary.png`.

### `edaSimDif.py`

This script performs Exploratory Data Analysis (EDA) on the 'Difference' (Actual - Simulated) target variable, providing insights into the simulation's bias and distribution.

* **Input Data**: `feature_importance_sims.csv` (path configured in `config.INPUT_CSV_PATH`).

* **Key Operations**:

    * Calculates and prints the mean, median, standard deviation, and skewness of the 'Difference' variable.

    * Provides an interpretation of the simulation's tendency to under/overestimate damage based on the mean difference and skewness.

    * Generates a histogram and a boxplot of the 'Difference' distribution.

* **Output Data**: `difference_analysis_plots.png` (saved to the `eda_results` directory).

### `featureImportance.py`

This script identifies the most influential features for predicting the target variable using mutual information.

* **Input Data**: `feature_importance_sims.csv` (path configured in `config.INPUT_CSV_PATH`).

* **Key Operations**:

    * Separates features (X) and target (y).

    * Preprocesses data for mutual information calculation by imputing NaN values in numeric columns with the median and label encoding object columns.

    * Calculates mutual information scores between each feature and the target.

    * Displays and plots the top `N_TOP_FEATURES` (set to 15 in the script) most influential features.

* **Output Data**: `feature_importance_plot.png` (saved to the `eda_results` directory).

### `4_modeling.py`

This script is responsible for feature selection through clustering and benchmarking various regression models on the prepared data.

* **Input Data**: `X_train_processed.pkl`, `y_train.pkl`, `X_test_processed.pkl`, `y_test.pkl` (from `2_dataPreprocessing.py`).

* **Key Operations**:

    * **Feature Selection via Clustering**: Uses hierarchical clustering on the Cramer's V association matrix of features, testing various `CLUSTERING_THRESHOLDS_TO_TEST` (e.g., `None` for all features) to group highly correlated features and select a representative from each cluster.

    * **Model Benchmarking**: Evaluates a range of regression models defined in `config.MODELS_TO_BENCHMARK`, including Linear Regression, Ridge, Lasso, ElasticNet, SVR, Decision Tree, Random Forest, Gradient Boosting, Hist Gradient Boosting, XGBoost, and LightGBM.

    * **Hyperparameter Tuning**: Performs `GridSearchCV` with 5-fold cross-validation (`N_SPLITS_CV`) to find the best hyperparameters for each model, optimizing for `neg_mean_squared_error` (`GRIDSEARCH_SCORING_METRIC`).

    * **Evaluation Metrics**: Calculates R2 Score, Mean Squared Error (MSE), and Mean Absolute Error (MAE) on the test set for each model.

* **Output Data**: `clustering_performance_detailed_results.csv` (saved to `config.DETAILED_RESULTS_CSV`), `best_estimators_per_combo.pkl` (saved to `config.BEST_ESTIMATORS_PATH`). Also generates `model_comparison_r2_score.png` and `actual_vs_predicted_*.png` plots for the top models.

### `shap_interpretation.py`

This script calculates and visualizes SHAP (SHapley Additive exPlanations) values for the top-performing models identified in the pipeline, providing insights into feature contributions and their impact on predictions.

* **Input Data**: `X_train_processed.pkl`, `X_test_processed.pkl`, `best_estimators_per_combo.pkl`, `preprocessor.pkl`, `clustering_performance_detailed_results.csv`, and `cleaned_data.csv`.

* **Key Operations**:

    * Identifies the best model based on the highest 'Test R2 Score' from the detailed performance results.

    * Calculates SHAP values for a sample of the test data using `shap.TreeExplainer`.

    * Generates a global feature importance bar chart and a detailed beeswarm plot from the SHAP values.

    * Visualizes the relationship between the top 5 original features (identified by SHAP) and the 'Difference' target variable using scatter plots (for numeric features) or strip/box plots (for categorical features).

* **Output Data**: `shap_summary_details.csv`, `shap_summary_bar.png`, `shap_beeswarm_plot.png` (saved to `config.SHAP_RESULTS_DIR`), and `relationship_*.png` plots for top features (saved to the `top_feature_plots` directory).

### `8_generate_report.py`

This script consolidates all generated visual outputs (plots and figures) from the analysis pipeline into a single PDF report.

* **Input Data**: All `.png` and `.jpg` image files found in specified directories (including `clustering_performance_results`, `shap_results_top_performers`, `eda_results`, `top_feature_plots`, and the current directory).

* **Key Operations**:

    * Searches predefined directories for image files generated during the pipeline run.

    * Uses the `FPDF` library to create a multi-page PDF document.

    * Includes a title page, a table of contents, and adds each found image on a new page, automatically resizing and centering them while determining optimal page orientation.

* **Output Data**: `pipeline_visual_report.pdf`

## Utility Scripts (Not part of the main sequential pipeline but useful)

### `RFE_tester.py`

This script is a helper tool for performing Recursive Feature Elimination with Cross-Validation (RFECV) for regression tasks. It is typically run independently to inform the `N_FEATURES_TO_SELECT` parameter in `config.py`.

* **Purpose**: To determine the optimal number of features for a regression model by evaluating performance as features are recursively removed.

* **Input Data**: `X_train_processed.pkl`, `y_train.pkl`, and `preprocessor.pkl`.

* **Key Operations**: Sets up `RFECV` with a `RandomForestRegressor` and `KFold` cross-validation, and runs the feature selection process.

* **Output Data**: `rfe_performance_vs_features.png`, which visualizes the cross-validated score against the number of selected features.

### `featureVis.py`

This is a flexible script designed for generating custom scatter plots.

* **Purpose**: Allows users to visualize relationships between two columns, with optional color encoding (`HUE_COLUMN`) and point size variation (`SIZE_COLUMN`).

* **Input Data**: `cleaned_data_latlong.csv` (configurable).

* **Usage**: Requires manual modification of `X_COLUMN`, `Y_COLUMN`, and optional `HUE_COLUMN`, `SIZE_COLUMN` variables within the script itself to specify the desired visualization.

* **Output Data**: Scatter plots (saved to the `data_visualizations` directory).

## Notes on Development

* The `6_deltaAccuracy.py` and original `7_shap.py` scripts were part of an earlier classification pipeline or exploration but are currently commented out in `main.py` and not actively used. This is due to them being "not meaningful bc low accuracy models" in that specific context.

* The `simulated_yishuang` directory contains experimental or older versions of some scripts (`modelGenReg_sim.py`, `mutualInfo-corr.py`, `simulatedFeatures.py`) that informed the current pipeline design. The core pipeline described above uses the main scripts in the repository's root.