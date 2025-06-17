import os
import logging
import sys
import time
from fpdf import FPDF
from PIL import Image
import config  # Import the configuration file


# --- Logging Configuration ---
def setup_logging(log_file=config.PIPELINE_LOG_PATH):
    """Sets up logging to append to the main pipeline log file."""
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
    """Custom PDF class to define a standard header and footer."""

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Mayfield Features - Visual Analysis Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_image_to_page(self, image_path, title):
        """Adds a new page with a title and a resized image."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            logging.error(f"Could not open image {image_path}: {e}")
            return

        orientation = 'L' if width > height else 'P'
        self.add_page(orientation=orientation)

        page_width = self.w - 20  # Page width with margins

        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

        img_aspect = width / height
        w = page_width
        h = w / img_aspect

        x = (self.w - w) / 2
        self.image(image_path, x=x, w=w, h=h)


# --- Main Report Generation Script ---
def main():
    logging.info(f"--- Starting Script: 8_generate_report.py ---")
    os.makedirs(config.REPORT_DIR, exist_ok=True)

    # Find all generated plots
    image_files = []
    for directory in [config.BASE_RESULTS_DIR, config.SHAP_RESULTS_DIR, '.']:  # Add '.' to catch plots in root
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(directory, f))

    if not image_files:
        logging.warning("No image files found to generate a report.")
        return

    logging.info(f"Found {len(image_files)} images to include in the report.")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "Visual Analysis Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(20)

    # Table of Contents
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Table of Contents", 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    for i, image_path in enumerate(sorted(image_files)):
        title = os.path.basename(image_path).replace('.png', '').replace('_', ' ').title()
        pdf.cell(0, 8, f"{i + 1}. {title}", 0, 1, 'L')

    # Add each image to a new page
    for image_path in sorted(image_files):
        logging.info(f"  Adding {image_path} to report...")
        title = os.path.basename(image_path).replace('.png', '').replace('_', ' ').title()
        pdf.add_image_to_page(image_path, title)

    pdf.output(config.REPORT_FILENAME)
    logging.info(f"Report saved to {config.REPORT_FILENAME}")
    logging.info(f"--- Finished Script: 8_generate_report.py ---")


if __name__ == '__main__':
    main()
