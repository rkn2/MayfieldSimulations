import pandas as pd
import numpy as np
import os
import joblib
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns  # Still useful for palettes
import re
import ast  # For safely evaluating string representations of dictionaries

# --- Scikit-learn ---
from sklearn.metrics import (
    accuracy_score, f1_score, make_scorer
)
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier

# --- Clustering & Association ---
from dython.nominal import associations
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# --- Optional Imports for Models ---
try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# --- Configuration ---
DATA_DIR = 'processed_ml_data'
BASE_RESULTS_DIR = 'cluster_importance_results_loaded_params'
BEST_PARAMS_CSV_PATH = os.path.join(DATA_DIR, 'model_tuned_cv_results.csv')

TRAIN_X_PATH = os.path.join(DATA_DIR, 'X_train_processed.pkl')
TRAIN_Y_PATH = os.path.join(DATA_DIR, 'y_train.pkl')
TEST_X_PATH = os.path.join(DATA_DIR, 'X_test_processed.pkl')
TEST_Y_PATH = os.path.join(DATA_DIR, 'y_test.pkl')

# --- USER-DEFINED CLUSTERING THRESHOLD ---
USER_DEFINED_CLUSTERING_THRESHOLD = 0.7
N_TOP_CLUSTERS_TO_PLOT = 10
RANDOM_STATE = 42

CLUSTERING_LINKAGE_METHOD = 'average'
SCORING_METRIC_FOR_IMPORTANCE = 'f1_weighted'

MODELS_TO_EVALUATE = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
if XGB_AVAILABLE:
    MODELS_TO_EVALUATE["XGBoost"] = xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False)
if LGBM_AVAILABLE:
    MODELS_TO_EVALUATE["LightGBM"] = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1)

# --- PROPOSED CLUSTER LABELS (Mapping Representative Feature to Descriptive Label) ---
# This dictionary should be comprehensive for the chosen threshold.
PROPOSED_CLUSTER_LABELS = {
    "cat__construction_type_u_maonsry_un": "Unspecified Masonry Construction Details",
    "cat__wall_cladding_u_vinyl_panel": "Vinyl Panel Cladding & Specific Retrofit Type",
    "cat__building_use_during_tornado_reisdential": "Residential Use During Tornado",  # Simplified
    "cat__construction_type_u_un": "Unspecified Construction & Wall System",
    "cat__soffit_present_u_un": "Rear Large Door Presence & Unspecified Soffit/Metal Roof Cover",
    "cat__door_present_s_no": "South-Facing Door Presence (Binary)",
    "num__parking_storage_facility": "Storage Occupancy & Facility Characteristics",
    "cat__door_present_n_no": "North-Facing/Front Door Presence & Stone Wall Cladding",
    "num__owner_government": "Government-Owned Utility Buildings with Specific Construction/Retrofit Details",
    "cat__occupany_u_religious": "Religious Occupancy with Stone/Other Cladding",
    "cat__masonry_leaves_triple_leaf": "Triple Leaf Masonry & Unspecified Wooden Wall Substrate",
    "cat__structural_wall_system_u_masonry_block_reinforced": "Reinforced Masonry Block Wall System with Steel/Metal Components",
    "cat__r2wall_attachment_u_metal_straps": "Roof-to-Wall Attachment (Metal Straps/Unspecified) & Right Large Door Presence",
    "cat__construction_type_u_steel_frame": "Steel Frame Construction with Concrete Roof Substrate & Wood Horizontal Material",
    "cat__front_elevation_orientation_n": "North-Facing Front Elevation with Continuous Wooden Sheathing Substrate",
    "cat__mwfrs_u_roof_un": "Building Height/Storey & Roof Characteristics",
    "cat__occupany_u_factory": "Factory Occupancy & Building Dimensions with Unspecified Masonry/Cladding",
    "num__wall_fenestration_per_e": "East Wall Fenestration Percentage",  # Simplified
    "num__const_material_h_othr": "Unreinforced Masonry Block Wall & Other Horizontal Construction Material",
    "num__const_material_v_rf_conc": "Diverse Heavy Construction Materials (Stone, Concrete, Masonry) & Wall Anchorage",
    "cat__foundation_type_u_concrete_piers": "Concrete Pier Foundation & Wooden Truss Roof with Back Wall Fenestration Protection Status",
    "cat__wall_fenesteration_protection_front_no": "Fenestration Protection Details & Religious/Assembly Occupancy Characteristics",
    "cat__front_elevation_orientation_s": "South or West Front Elevation Orientation",
    "cat__retrofit_present_u_yes": "Retrofit Presence & Type with Specific Building Setting/Components",
    "num__const_material_v_brick": "Brick Construction Material (Vertical/Horizontal) & Wood Vertical Material",
    "cat__masonry_leaves_double_leaf": "Masonry Leaf Type (Single/Double) & Brick Cladding",
    "cat__roof_substrate_type_u_dimensional_lumber": "Roof Substrate Type (Dimensional Lumber or Unspecified)",
    "cat__construction_type_u_steel_frame_masonry_shear_wall": "Steel Frame with Masonry Shear Wall Construction Details",
    "cat__soffit_type_u_not_applicable": "Soffit Presence/Type & Slab-on-Grade Foundation",
    "num__roof_slope_u": "General Building Typology & Use (Residential/Business Focus)",
    "cat__buidling_use_before_tornado_not_in_use": "Unoccupied/Not-in-Use Buildings with Specific Roof/Soffit Details",
    "num__longitude": "Geographic Location & Stone Foundation",
    "num__wall_fenesteration_per_back": "Back Wall Fenestration Percentage & End-of-Row Urban Setting",
    "num__wall_fenesteration_per_front": "Front/West Wall Fenestration & Stone Horizontal Material with Complex Roof/Vinyl Soffit",
    "num__random_feature": "Random Feature (Isolated)",  # Simplified
    "cat__buidling_use_before_tornado_educational": "NGO-Owned Educational Wood Frame Buildings (Year Built Unspecified)",
    "cat__occupany_u_educational": "Educational Occupancy with Recent Retrofit & Unspecified Moment Frame",
    "num__single_unit": "Building Unit Type (Single/Multiple) & Masonry Stem Wall Foundation",
    "cat__occupany_u_museum": "Museum Occupancy with Early Retrofit Year",
    "cat__mwfrs_u_wall_wall_diaphragm_masonry": "Wall Diaphragm Material (Masonry/Wood) & Solid Brick Wythe System",
    "cat__retrofit_year_u_un": "Retrofit Year (Unspecified or Very Old) & Weatherboard Cladding",
    "cat__roof_cover_u_slate": "Slate Roof Cover",  # Simplified
    "cat__retrofit_type_u_steel_bracing": "Steel Bracing Retrofit Type",  # Simplified
    "cat__construction_type_u_steel_light_frame": "Steel Light Frame Construction",  # Simplified
    "cat__foundation_type_u_stone_un": "Unspecified Stone Foundation",  # Simplified
    "num__sub_national_heritage__list": "Sub-National Heritage List Status",  # Simplified
    "cat__wall_cladding_u_wood_ot": "Wood/Other Wall Cladding",  # Simplified
    "cat__retrofit_type_u_reinforced_masonry_awning_window_covers": "Specific Retrofit (Reinforced Masonry Awning/Window Covers)",
    # Simplified
    "num__const_material_h_mud": "Horizontal Mud Construction Material",  # Simplified
    "num__const_material_v_mud": "Vertical Mud Construction Material",  # Simplified
    "cat__building_in_use_during_tornado_yes": "Building in Use During Tornado (Yes)"  # Simplified
}


# --- Helper Functions ---
def load_pickle_data(file_path, description="data"):
    print(f"Loading {description} from {file_path}...")
    try:
        data = joblib.load(file_path)
        print(f"  Successfully loaded. Shape: {data.shape if hasattr(data, 'shape') else 'N/A (Series)'}")
        return data
    except FileNotFoundError:
        print(f"Error: {description} file not found at {file_path}.")
        return None
    except Exception as e:
        print(f"Error loading {description} from {file_path}: {e}")
        return None


def sanitize_feature_names_df(df):
    if not isinstance(df, pd.DataFrame): return df
    original_cols = df.columns.tolist()
    new_cols = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in original_cols]
    new_cols = [re.sub(r'[\[\]<]', '_', col) for col in new_cols]
    final_cols = []
    counts = {}
    for col_name in new_cols:
        if col_name in counts:
            counts[col_name] += 1
            final_cols.append(f"{col_name}_{counts[col_name]}")
        else:
            counts[col_name] = 0
            final_cols.append(col_name)
    df.columns = final_cols
    return df


def calculate_association_df(dataframe):
    print("  Calculating association matrix for clustering...")
    for col in dataframe.columns:
        if dataframe[col].dtype == 'bool': dataframe[col] = dataframe[col].astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assoc_results = associations(dataframe, nom_nom_assoc='cramer', compute_only=True, mark_columns=False,
                                     clustering=False, plot=False)
    association_dataframe = assoc_results['corr']
    print("  Association matrix calculated.")
    return association_dataframe.fillna(0)


def get_selected_features_by_clustering(original_df, distance_thresh, linkage_meth):
    feature_names_list = original_df.columns.tolist()
    if len(feature_names_list) <= 1:
        return feature_names_list

    assoc_df = calculate_association_df(original_df.copy())

    print(f"  Performing hierarchical clustering linkage ({linkage_meth})...")
    distance_mat = 1 - np.abs(assoc_df.values)
    np.fill_diagonal(distance_mat, 0)
    distance_mat = (distance_mat + distance_mat.T) / 2
    condensed_dist_mat = squareform(distance_mat, checks=False)

    if condensed_dist_mat.shape[0] == 0:
        print("  Warning: Condensed distance matrix is empty with multiple features. Returning all features.")
        return feature_names_list

    linked = hierarchy.linkage(condensed_dist_mat, method=linkage_meth)
    print(f"  Forming flat clusters with distance threshold: {distance_thresh}...")
    cluster_labels_arr = hierarchy.fcluster(linked, t=distance_thresh, criterion='distance')
    num_unique_clusters = len(np.unique(cluster_labels_arr))

    selected_representatives_list = []
    for i in range(1, num_unique_clusters + 1):
        cluster_member_indices_list = [idx for idx, label in enumerate(cluster_labels_arr) if label == i]
        if not cluster_member_indices_list: continue
        if len(cluster_member_indices_list) == 1:
            selected_representatives_list.append(feature_names_list[cluster_member_indices_list[0]])
        else:
            cluster_assoc_submat = assoc_df.iloc[cluster_member_indices_list, cluster_member_indices_list]
            sum_abs_assoc_arr = np.abs(cluster_assoc_submat.values).sum(axis=1)
            rep_local_idx = np.argmax(sum_abs_assoc_arr)
            rep_original_idx = cluster_member_indices_list[rep_local_idx]
            selected_representatives_list.append(feature_names_list[rep_original_idx])
    return sorted(list(set(selected_representatives_list)))


def load_best_parameters(params_csv_path):
    print(f"\nLoading best parameters from: {params_csv_path}")
    try:
        params_df = pd.read_csv(params_csv_path)
        model_best_params = {}
        for _, row in params_df.iterrows():
            model_name = row['Model']
            try:
                params_str = row['Best Params']
                if pd.isna(params_str) or params_str.lower() == 'n/a' or params_str == '{}':
                    model_best_params[model_name] = {}
                    print(f"  No valid parameters found for {model_name}, will use defaults.")
                else:
                    best_params_dict = ast.literal_eval(params_str)
                    model_best_params[model_name] = best_params_dict
            except Exception as e:
                print(f"  Warning: Could not parse params for {model_name}: {e}. Using default params.")
                model_best_params[model_name] = {}
        print("Best parameters loaded.")
        return model_best_params
    except FileNotFoundError:
        print(f"Error: Best parameters CSV file not found at {params_csv_path}.")
        return None
    except Exception as e:
        print(f"Error loading best parameters: {e}")
        return None


# --- Main Orchestration ---
def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    current_clustering_threshold = USER_DEFINED_CLUSTERING_THRESHOLD
    results_subdir = os.path.join(BASE_RESULTS_DIR, f"thresh_{current_clustering_threshold}")
    os.makedirs(results_subdir, exist_ok=True)

    print(
        f"Starting Multi-Model Cluster Importance (with Loaded Params & Descriptive Labels) for THRESHOLD = {current_clustering_threshold}")

    all_best_params = load_best_parameters(BEST_PARAMS_CSV_PATH)
    if all_best_params is None:
        print("Could not load best parameters. Exiting.")
        return

    X_train_orig = load_pickle_data(TRAIN_X_PATH, "original X_train")
    y_train_orig = load_pickle_data(TRAIN_Y_PATH, "original y_train")
    X_test_orig = load_pickle_data(TEST_X_PATH, "original X_test")
    y_test_orig = load_pickle_data(TEST_Y_PATH, "original y_test")

    if X_train_orig is None or y_train_orig is None or X_test_orig is None or y_test_orig is None:
        print("Exiting due to missing data.")
        return

    X_train_sanitized = sanitize_feature_names_df(
        X_train_orig.copy() if isinstance(X_train_orig, pd.DataFrame) else pd.DataFrame(X_train_orig))
    X_test_sanitized = sanitize_feature_names_df(
        X_test_orig.copy() if isinstance(X_test_orig, pd.DataFrame) else pd.DataFrame(X_test_orig))

    if isinstance(X_train_sanitized, pd.DataFrame) and isinstance(X_test_sanitized, pd.DataFrame):
        train_cols = X_train_sanitized.columns
        X_test_sanitized = X_test_sanitized.reindex(columns=train_cols, fill_value=0)

    y_train_ravel = y_train_orig.values.ravel() if isinstance(y_train_orig,
                                                              (pd.Series, pd.DataFrame)) else y_train_orig.ravel()
    y_test_ravel = y_test_orig.values.ravel() if isinstance(y_test_orig,
                                                            (pd.Series, pd.DataFrame)) else y_test_orig.ravel()

    if not isinstance(X_train_sanitized, pd.DataFrame) or X_train_sanitized.shape[1] <= 1:
        print(
            "Warning: X_train is not a suitable DataFrame for clustering or has too few features. Using all features.")
        selected_features_representatives = X_train_sanitized.columns.tolist() if isinstance(X_train_sanitized,
                                                                                             pd.DataFrame) else [
            f"feature_{i}" for i in range(X_train_sanitized.shape[1])]
    else:
        print(f"\nPerforming feature selection with clustering threshold: {current_clustering_threshold}...")
        selected_features_representatives = get_selected_features_by_clustering(
            # This returns representative feature names
            X_train_sanitized,
            current_clustering_threshold,
            CLUSTERING_LINKAGE_METHOD
        )

    if not selected_features_representatives:
        print("Error: No features selected by clustering. Cannot proceed.")
        return

    num_selected_features = len(selected_features_representatives)
    print(f"Number of features (clusters) selected: {num_selected_features}")
    if num_selected_features == 0:
        print("No features selected, stopping analysis.")
        return

    if isinstance(X_train_sanitized, pd.DataFrame):
        X_train_selected = X_train_sanitized[selected_features_representatives]
        X_test_selected = X_test_sanitized[selected_features_representatives]
    else:
        print("Warning: Feature selection on NumPy array requires careful index mapping.")
        X_train_selected = X_train_sanitized
        X_test_selected = X_test_sanitized

    print("\n--- Training models with pre-loaded best parameters ---")
    trained_models = {}

    for model_name, model_template in MODELS_TO_EVALUATE.items():
        print(f"  Training model: {model_name}...")
        model_params = all_best_params.get(model_name, {})
        current_model = MODELS_TO_EVALUATE[model_name]
        try:
            valid_params = {k: v for k, v in model_params.items() if hasattr(current_model, k)}
            if model_name == "XGBoost" and 'eval_metric' in model_params and 'eval_metric' not in valid_params:
                pass
            if valid_params:
                current_model.set_params(**valid_params)
                print(f"    Set parameters for {model_name}: {valid_params}")
            else:
                print(
                    f"    Using default parameters for {model_name} as no valid pre-tuned params were found/applicable.")
            current_model.fit(X_train_selected, y_train_ravel)
            trained_models[model_name] = current_model
            print(f"    Successfully trained {model_name}.")
        except Exception as e:
            print(f"    ERROR: Model {model_name} failed during training with loaded params: {e}")
            trained_models[model_name] = None
            continue

    if not any(trained_models.values()):
        print("No models successfully trained. Cannot proceed with feature importance.")
        return

    all_model_importances_list = []
    print(f"\nCalculating Permutation Importance for selected features using all trained models...")

    if SCORING_METRIC_FOR_IMPORTANCE == 'f1_weighted':
        perm_scorer = make_scorer(f1_score, average='weighted', zero_division=0)
    elif SCORING_METRIC_FOR_IMPORTANCE == 'accuracy':
        perm_scorer = 'accuracy'
    else:
        print(
            f"Warning: Permutation importance using default or potentially misconfigured scorer for '{SCORING_METRIC_FOR_IMPORTANCE}'.")
        perm_scorer = SCORING_METRIC_FOR_IMPORTANCE

    for model_name, model_instance in trained_models.items():
        if model_instance is None:
            print(f"  Skipping permutation importance for {model_name} as it was not successfully trained.")
            continue
        print(f"  Calculating for model: {model_name}...")
        try:
            perm_importance_result = permutation_importance(
                model_instance,
                X_test_selected,
                y_test_ravel,
                scoring=perm_scorer,
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            # Map representative feature name to descriptive cluster label
            for i, rep_feature_name in enumerate(selected_features_representatives):
                descriptive_label = PROPOSED_CLUSTER_LABELS.get(rep_feature_name,
                                                                rep_feature_name)  # Fallback to rep_feature if no label
                all_model_importances_list.append({
                    'Cluster Label': descriptive_label,  # Use descriptive label
                    'Representative Feature': rep_feature_name,  # Keep original representative for reference
                    'Model': model_name,
                    'Importance (Mean Drop in Score)': perm_importance_result.importances_mean[i],
                    'Importance (Std Dev)': perm_importance_result.importances_std[i]
                })
        except Exception as e_perm:
            print(f"    Error calculating permutation importance for {model_name}: {e_perm}")

    if not all_model_importances_list:
        print("No permutation importance results to visualize.")
        return

    all_importances_df = pd.DataFrame(all_model_importances_list)

    # Determine Top N clusters based on max importance of the *descriptive label*
    max_importance_per_cluster_label = all_importances_df.groupby('Cluster Label')[
        'Importance (Mean Drop in Score)'].max()
    top_n_cluster_labels = max_importance_per_cluster_label.sort_values(ascending=False).head(
        N_TOP_CLUSTERS_TO_PLOT).index.tolist()

    top_n_importances_df = all_importances_df[all_importances_df['Cluster Label'].isin(top_n_cluster_labels)]

    # Ensure consistent ordering for plotting based on the descriptive labels
    top_n_importances_df['Cluster Label'] = pd.Categorical(
        top_n_importances_df['Cluster Label'],
        categories=top_n_cluster_labels,
        ordered=True
    )
    top_n_importances_df = top_n_importances_df.sort_values('Cluster Label')

    print("\n--- Top Clustered Feature Importances (Across All Models) ---")

    importance_csv_path = os.path.join(results_subdir,
                                       f"multi_model_cluster_importances_thresh_{current_clustering_threshold}.csv")
    all_importances_df.to_csv(importance_csv_path, index=False)
    print(f"\nFull multi-model cluster importances saved to: {importance_csv_path}")

    # --- Plotting with Matplotlib for Grouped Bars with Error Bars ---
    plot_df = top_n_importances_df.pivot(
        index='Cluster Label',  # Use descriptive label for index
        columns='Model',
        values='Importance (Mean Drop in Score)'
    )
    plot_df_std = top_n_importances_df.pivot(
        index='Cluster Label',  # Use descriptive label for index
        columns='Model',
        values='Importance (Std Dev)'
    )

    # Reindex to ensure the order of features in plot_df matches top_n_cluster_labels
    plot_df = plot_df.reindex(top_n_cluster_labels)
    plot_df_std = plot_df_std.reindex(top_n_cluster_labels)

    n_models = len(plot_df.columns)
    n_features_to_plot = len(plot_df.index)  # Now based on cluster labels

    bar_height = 0.8 / n_models
    index = np.arange(n_features_to_plot)

    fig, ax = plt.subplots(
        figsize=(15, max(8, n_features_to_plot * 0.7 * n_models * 0.3)))  # Adjusted width for potentially longer labels

    colors = sns.color_palette('viridis', n_colors=n_models)

    for i, model_name_plot in enumerate(plot_df.columns):
        means = plot_df[model_name_plot].values
        stds = plot_df_std[model_name_plot].values

        bar_positions = index - (0.8 - bar_height) / 2 + i * bar_height

        ax.barh(
            bar_positions,
            means,
            bar_height * 0.9,
            xerr=stds,
            label=model_name_plot,
            color=colors[i],
            capsize=3,
            alpha=0.8
        )

    ax.set_xlabel(
        f"Mean Drop in Test {SCORING_METRIC_FOR_IMPORTANCE.replace('_', ' ').title()} (Permutation Importance)",
        fontsize=12)
    ax.set_ylabel("Descriptive Cluster Label", fontsize=12)  # Updated Y-axis label
    ax.set_title(
        f'Top {min(N_TOP_CLUSTERS_TO_PLOT, n_features_to_plot)} Most Important Clusters (Thresh={current_clustering_threshold})',
        fontsize=14)
    ax.set_yticks(index)
    ax.set_yticklabels(plot_df.index)  # Use descriptive cluster labels for y-tick labels
    ax.invert_yaxis()
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 0.83, 1])  # Adjusted right margin for legend

    plot_save_path = os.path.join(results_subdir,
                                  f"top_multi_model_cluster_importances_desc_labels_thresh_{current_clustering_threshold}.png")
    plt.savefig(plot_save_path)
    print(f"Multi-model importance plot saved to: {plot_save_path}")
    plt.show()

    print("\nMulti-model cluster importance analysis with descriptive labels finished.")


if __name__ == '__main__':
    main()
