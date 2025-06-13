# Data Analysis Pipeline Details

This repository implements a data analysis pipeline to process, analyze, and report on tornado damage data. Each script in the pipeline serves a specific purpose, handling data from initial cleaning to advanced model interpretability and report generation.

### Pipeline Details

The pipeline consists of the following scripts, executed sequentially by `main.py`:

#### 1. `1_dataCleaning.py`

This script focuses on cleaning the raw input data, transforming it into a more usable format for subsequent machine learning tasks.

* **Input Data**: `QuadState_Tornado_DataInputv2.csv`
* **Key Choices/Operations**:
    * **Initial Row Filtering**: Drops rows where the `degree_of_damage_u` column (target variable) is missing.
    * **Low Variation Column Removal**: Removes columns with very few unique values (less than `LOW_VARIATION_THRESHOLD`, which is set to 1) to eliminate non-informative features.
    * **Column Dropping by Keywords**: Drops columns containing specified keywords (e.g., 'photos', 'details', 'prop_', '_unc') which are likely irrelevant or redundant.
    * **Specific Column Removal**: Removes a predefined list of specific columns that are not needed for analysis (e.g., 'completed\_by', 'damage\_status', 'notes').
    * **Older Version Column Removal**: If both an original (`_o`) and an updated (`_u`) version of a column exist, the older (`_o`) version is dropped.
    * **Percentage and Directional Column Handling**: Configurable to either `keep` or `drop` columns containing `_per_` (percentage) or ending with directional suffixes (`_n`, `_s`, `_e`, `_w`).
    * **Specific Value Replacements**: Replaces non-numeric placeholder values (e.g., 'un', 'not\_applicable') with numerical equivalents (e.g., 0) in specific columns (`wall_thickness`, `overhang_length_u`, `parapet_height_m`), followed by conversion to numeric type.
    * **Missing Value Imputation**: Numeric columns are imputed with their mean, while object/string columns have missing or empty values replaced with 'un'.
* **Output Data**: `cleaned_data_latlong.csv`

#### 2. `2_dataPreprocessing.py`

This script prepares the cleaned data for machine learning by handling the target variable, applying class reduction, balancing the dataset, and performing feature transformations.

* **Input Data**: `cleaned_data_latlong.csv` (output from `1_dataCleaning.py`)
* **Key Choices/Operations**:
    * **Damage Level 0 Subsampling**: Optionally reduces the number of rows for damage level 0 to `SUBSAMPLE_DAMAGE_0` (set to 50) to address potential class imbalance, primarily if class 0 is not remapped by class reduction.
    * **Target Variable Separation**: Extracts the `degree_of_damage_u` column as the target variable (`y`).
    * **Class Reduction Strategy**: Applies a class reduction strategy (`REDUCE_CLASSES_STRATEGY`, set to 'B') to simplify the target variable's categories. Strategy 'B' maps damage levels 1, 2, 3, 4 to a single 'Repairable (1)' category, while 0 remains 'Undamaged (0)' and 5 becomes 'Demolished (2)'.
    * **Feature Filtering**: Removes features from the input data (`X`) that contain certain keywords (e.g., 'damage', 'status\_u', 'exist', 'demolish') to avoid data leakage or redundant information related to the target.
    * **Data Splitting**: Divides the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`) using a `TEST_SIZE` of 0.2 and stratification based on the target variable's distribution.
    * **Feature Preprocessing**:
        * **Numeric Features**: Standardized using `StandardScaler`.
        * **Categorical Features**: Converted to numerical format using `OneHotEncoder`.
    * **Data Balancing**: Applies `SMOTE` (Synthetic Minority Over-sampling Technique) to the training data to address class imbalance, oversampling minority classes to balance the dataset.
* **Output Data**: `X_train_processed.pkl`, `y_train.pkl`, `X_test_processed.pkl`, `y_test.pkl`, and `preprocessor.pkl` (for transforming new data) saved in the `processed_ml_data` directory.

#### 3. `4_clustering.py`

This script is responsible for feature selection through clustering and benchmarking various machine learning models on the prepared data.

* **Input Data**: `X_train_processed.pkl`, `y_train.pkl`, `X_test_processed.pkl`, `y_test.pkl` (from `2_dataPreprocessing.py`)
* **Key Choices/Operations**:
    * **Feature Selection via Clustering**: Uses hierarchical clustering on the Cramer's V association matrix of features. It tests various `CLUSTERING_THRESHOLDS_TO_TEST` (0.1 to 1.0, plus None for all features) to group highly correlated features and select a representative from each cluster. The `CLUSTERING_LINKAGE_METHOD` is 'average'.
    * **Model Benchmarking**: Evaluates a range of classification models, including:
        * Logistic Regression
        * Decision Tree
        * Random Forest
        * Gradient Boosting
        * Hist Gradient Boosting
        * K-Nearest Neighbors (KNN)
        * XGBoost (if available)
        * LightGBM (if available)
        * Ordinal Logistic (AT), Ordinal Ridge, Ordinal LAD (if `mord` available for ordinal classification)
    * **Hyperparameter Tuning**: Performs `GridSearchCV` with 5-fold cross-validation (`N_SPLITS_CV`) to find the best hyperparameters for each model, optimizing for `f1_weighted` score (`GRIDSEARCH_SCORING_METRIC`).
    * **Evaluation Metrics**: Calculates mean cross-validated scores for accuracy, f1-weighted, f1-macro, precision-weighted, and recall-weighted on the training set, and then evaluates these metrics on the test set.
* **Output Data**: `clustering_performance_detailed_results.csv` containing all model performance metrics and best parameters for each feature set. Confusion matrix plots for the top 5 performing models are also generated and saved in the `clustering_performance_results` directory.

#### 4. `6_deltaAccuracy.py`

This script analyzes feature importance using permutation importance for high-performing model-feature set combinations.

* **Input Data**: `X_train_processed.pkl`, `y_train.pkl`, `X_test_processed.pkl`, `y_test.pkl` and `clustering_performance_detailed_results.csv` (from `4_clustering.py`)
* **Key Choices/Operations**:
    * **High-Performing Model Identification**: Loads the `clustering_performance_detailed_results.csv` and filters for models with a `Test F1 Weighted` score greater than `PERFORMANCE_THRESHOLD` (set to 0.8).
    * **Feature Clustering Recreation**: Recreates the specific clustered feature sets used by each high-performing model using the same methodology as `4_clustering.py`.
    * **Permutation Importance Calculation**: For each high-performing model and its selected features, it calculates permutation importance on the test set (`X_test_selected`, `y_test_ravel`). It measures the mean drop in `f1_weighted` score after shuffling a feature's values, repeated `N_PERMUTATION_REPEATS` times (set to 200).
    * **P-Value Calculation**: A p-value is calculated for each feature's importance to determine statistical significance, based on how many permutations resulted in importance less than or equal to zero.
    * **Visualization**: Generates a bar plot showing statistically significant feature clusters (p-value < `P_VALUE_THRESHOLD`, set to 0.05) and their mean importance drop across the top models.
* **Output Data**: A plot named `significant_cluster_importances.png` saved in the `cluster_importance_results_final` directory. Logs also contain detailed importance results.

#### 5. `7_shap.py`

This script calculates and visualizes SHAP (SHapley Additive exPlanations) values for the top-performing models identified in the pipeline, providing insights into feature contributions.

* **Input Data**: `X_train_processed.pkl`, `y_train.pkl`, `X_test_processed.pkl`, `y_test.pkl` and `clustering_performance_detailed_results.csv` (from `4_clustering.py`)
* **Key Choices/Operations**:
    * **High-Performing Model Identification**: Identifies models with a `Test F1 Weighted` score greater than `PERFORMANCE_THRESHOLD` (set to 0.8) from the detailed performance results.
    * **Model Retraining**: For each identified high-performing model, it re-initializes and retrains the model with its best parameters on the appropriate clustered feature set.
    * **SHAP Value Calculation**: Uses `shap.Explainer` to calculate SHAP values for a sample of the test data (`N_SHAP_SAMPLES`, set to 1000) against a background dataset from the training data (`N_BACKGROUND_SAMPLES`, set to 200).
    * **Visualization**: Generates two types of SHAP plots:
        * **Composite Bar Chart**: Shows the top `N_TOP_FEATURES_TO_PLOT` (set to 15) most important features based on mean absolute SHAP values across all analyzed high-performing models.
        * **Individual Beeswarm Plots**: Creates a beeswarm plot for each target class, showing the distribution of SHAP values for the top features, indicating how each feature impacts the model's output for that specific class.
* **Output Data**: SHAP summary plots (composite bar chart and individual beeswarm plots per class) are saved in the `shap_results_top_performers` directory.

#### 6. `8_generate_report.py`

This script consolidates all generated visual outputs (plots and figures) from the analysis pipeline into a single PDF report.

* **Input Data**: All `.png` and `.jpg` image files found in specified directories (`clustering_performance_results`, `cluster_importance_results_final`, `shap_results_top3`, `shap_results_top_performers`).
* **Key Choices/Operations**:
    * **Image Collection**: Recursively searches predefined directories for image files (`.png`, `.jpg`, `.jpeg`).
    * **PDF Generation**: Uses the `FPDF` library to create a multi-page PDF document.
    * **Report Structure**: Includes a title page, a simple table of contents listing all included images, and then adds each image on a new page.
    * **Image Layout**: Images are automatically resized to fit the page and centered. Page orientation (portrait or landscape) is determined based on the image's aspect ratio to ensure optimal display.
* **Output Data**: `pipeline_visual_report.pdf`, containing all the generated plots and figures from the analysis.