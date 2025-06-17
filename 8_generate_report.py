import os
import logging
import sys
import time
import pandas as pd
import joblib
from fpdf import FPDF
from PIL import Image
import config


# --- Logging Configuration ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )


setup_logging()


# --- Custom PDF Class ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Mayfield Features - Visual Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_image_to_page(self, image_path, title):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            logging.error(f"Could not open image {image_path}: {e}")
            return

        orientation = 'L' if width > height else 'P'
        self.add_page(orientation=orientation)
        page_width = self.w - 20

        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

        img_aspect = width / height
        w = page_width
        h = w / img_aspect
        x = (self.w - w) / 2
        self.image(image_path, x=x, w=w, h=h)


# --- Helper function to find files with detailed logging ---
def find_file(filename, search_dirs):
    """Searches for a file in a list of directories with verbose logging."""
    logging.info(f"  -> Searching for file: '{filename}'")
    for directory in search_dirs:
        path = os.path.join(directory, filename)
        logging.info(f"     -- Checking path: {path}")
        if os.path.exists(path):
            logging.info(f"     -- SUCCESS: Found at {path}")
            return path
    logging.warning(f"  -> FAILED: Could not find '{filename}' in any search directory.")
    return None


def get_original_feature_name_for_report(processed_name, cat_features):
    if processed_name.startswith('num__') or processed_name.startswith('remainder__'):
        return processed_name.split('__', 1)[1]
    if processed_name.startswith('cat__'):
        processed_suffix = processed_name.split('__', 1)[1]
        best_match = ''
        for cat_name in cat_features:
            if processed_suffix.startswith(cat_name + "_"):
                if len(cat_name) > len(best_match):
                    best_match = cat_name
        if best_match: return best_match
    return processed_name.split('__', 1)[-1]


def main():
    logging.info(f"--- Starting Script: 8_generate_report.py ---")
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # UPDATED: Search directories now include the new 'top_feature_plots'
    search_directories = [
        config.BASE_RESULTS_DIR,
        config.SHAP_RESULTS_DIR,
        'eda_results',
        'top_feature_plots',  # Added new directory
        '.'
    ]
    ordered_files = []

    # --- Build the ordered list of plots ---
    logging.info("Assembling report sections in logical order...")

    # 1. EDA on the target variable
    ordered_files.append('difference_analysis_plots.png')
    ordered_files.append('feature_importance_plot.png')

    # 2. RFE Plot
    ordered_files.append('rfe_performance_vs_features.png')

    # 3. Modeling Results
    ordered_files.append('model_comparison_r2_score.png')
    try:
        perf_df = pd.read_csv(config.DETAILED_RESULTS_CSV)
        top_5 = perf_df.sort_values(by='Test R2 Score', ascending=False).head(5)
        for _, row in top_5.iterrows():
            combo_key = f"{row['Model']}_{row['Feature Set Name']}"
            filename = f"actual_vs_predicted_{combo_key.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            ordered_files.append(filename)
    except FileNotFoundError:
        logging.warning(f"Could not find '{config.DETAILED_RESULTS_CSV}' to order model plots.")

    # 4. SHAP Interpretation
    ordered_files.append('shap_summary_bar.png')
    ordered_files.append('shap_beeswarm_plot.png')
    try:
        preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        shap_df = pd.read_csv(os.path.join(config.SHAP_RESULTS_DIR, 'shap_summary_details.csv'))
        original_cat_features = preprocessor.named_transformers_['cat'].feature_names_in_
        shap_df['Original_Feature'] = shap_df['Feature'].apply(
            lambda x: get_original_feature_name_for_report(x, original_cat_features))
        top_original_features = shap_df['Original_Feature'].unique()[:5]
        for feature in top_original_features:
            ordered_files.append(f"relationship_{feature}.png")
    except FileNotFoundError:
        logging.warning("Could not find SHAP summary to order relationship plots.")

    # --- Find full paths for the ordered files ---
    logging.info("\n--- Searching for plot files to include in report ---")
    found_image_files = []
    for filename in ordered_files:
        path = find_file(filename, search_directories)
        if path:
            found_image_files.append(path)

    if not found_image_files:
        logging.warning("No valid image files were found to generate a report.")
        return

    logging.info(f"\nFound {len(found_image_files)} images to include in the report.")

    # --- Generate PDF ---
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "Visual Analysis Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(20)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Table of Contents", 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    for i, image_path in enumerate(found_image_files):
        title = os.path.basename(image_path).replace('.png', '').replace('_', ' ').title()
        pdf.cell(0, 8, f"{i + 1}. {title}", 0, 1, 'L')

    for image_path in found_image_files:
        logging.info(f"  Adding {image_path} to report...")
        title = os.path.basename(image_path).replace('.png', '').replace('_', ' ').title()
        pdf.add_image_to_page(image_path, title)

    report_path = os.path.join(config.REPORT_DIR, config.REPORT_FILENAME)
    pdf.output(report_path)
    logging.info(f"Report saved to {report_path}")
    logging.info(f"--- Finished Script: 8_generate_report.py ---")


if __name__ == '__main__':
    main()
