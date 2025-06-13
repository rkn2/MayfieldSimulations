import os
import logging
import sys
import time
from fpdf import FPDF
from PIL import Image

# --- Configuration ---
LOG_FILE = 'pipeline.log'
REPORT_FILENAME = 'pipeline_visual_report.pdf'
# Define the directories where your plots are saved
DIRECTORIES_TO_SEARCH = [
    'clustering_performance_results',
    'cluster_importance_results_final',
    'shap_results_top3',
    'shap_results_top_performers'  # Added to include all possible output dirs
]
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg']


# --- Logging Configuration ---
def setup_logging():
    """Sets up logging to append to the main pipeline log file."""
    # This ensures that if the script is run standalone, it will still log to the console
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add a file handler to append to the main pipeline log
    logging.getLogger().addHandler(logging.FileHandler(LOG_FILE, mode='a'))


class PDF(FPDF):
    """
    Custom PDF class to define a standard header and footer for the report.
    """

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Mayfield Features - Visual Analysis Report', 0, 1, 'C')
        # Line break
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def add_image_to_page(self, image_path, title):
        """
        Adds a new page with a title and a resized image.
        Automatically handles page orientation based on image aspect ratio.
        """
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception as e:
            logging.error(f"Could not open or read image {image_path}: {e}")
            return

        # A4 page size in mm: 210 x 297
        page_width_p = 210 - 20  # Portrait page width with margin
        page_height_p = 297 - 40  # Portrait page height with margin

        # Decide on orientation based on image aspect ratio
        orientation = 'L' if width > height else 'P'
        self.add_page(orientation=orientation)

        page_width, page_height = (page_height_p, page_width_p) if orientation == 'L' else (page_width_p, page_height_p)

        # Add the chapter title on the new page
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

        # Calculate image aspect ratio and best fit
        img_aspect = width / height
        page_aspect = page_width / page_height

        if img_aspect > page_aspect:
            # Image is wider than the page, fit to width
            w = page_width
            h = w / img_aspect
        else:
            # Image is taller than the page, fit to height
            h = page_height
            w = h * img_aspect

        # Center the image
        x = (self.w - w) / 2
        self.image(image_path, x=x, w=w, h=h)


def find_image_files(directories):
    """Finds all image files in a list of directories and their subdirectories."""
    image_files = []
    for directory in directories:
        if not os.path.isdir(directory):
            logging.warning(f"Directory '{directory}' not found, skipping.")
            continue
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
    return sorted(image_files)


def main():
    """Main function to generate a PDF report from all generated figures."""
    setup_logging()
    logging.info(f"--- Starting Script: 8_generate_report.py ---")

    image_files = find_image_files(DIRECTORIES_TO_SEARCH)

    if not image_files:
        logging.warning("No image files found to generate a report. Exiting.")
        return

    logging.info(f"Found {len(image_files)} images to include in the report.")

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Create a Title Page ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "Visual Analysis Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
    pdf.ln(20)

    # --- Create a Table of Contents ---
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Table of Contents", 0, 1, 'L')
    pdf.set_font("Arial", '', 12)
    for i, image_file in enumerate(image_files):
        # Using a simple list for the table of contents
        pdf.cell(0, 8, f"{i + 1}. {os.path.basename(image_file)}", 0, 1, 'L')

    # --- Add each image to a new page in the PDF ---
    for image_path in image_files:
        logging.info(f"  Adding {image_path} to the report...")
        # Create a clean title from the filename
        title = os.path.basename(image_path).replace('.png', '').replace('_', ' ').title()
        pdf.add_image_to_page(image_path, title)

    pdf.output(REPORT_FILENAME)
    logging.info(f"\nSuccessfully generated visual report: {REPORT_FILENAME}")
    logging.info(f"--- Finished Script: 8_generate_report.py ---")


if __name__ == '__main__':
    main()
