import os
import subprocess
import logging

# --- Configuration ---
LOG_FILE = 'pipeline.log'
SCRIPTS_TO_RUN = [
    '1_dataCleaning.py',
    '2_dataPreprocessing.py',
    '4_clustering.py',
    '6_deltaAccuracy.py',
    '7_shap.py',
    '8_generate_report.py' # Added the new script
]


# --- Main Execution Function ---
def main():
    """
    Orchestrates the execution of the entire data analysis pipeline.
    """
    # --- Clean up previous log file ---
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Removed old log file: {LOG_FILE}")

    # --- Setup logging for this main script ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )

    logging.info("--- Starting the Data Analysis Pipeline ---")

    # --- Execute each script in sequence ---
    for script in SCRIPTS_TO_RUN:
        logging.info(f"--- Running: {script} ---")

        # Using subprocess.run for better control and error capturing
        result = subprocess.run(['python', script], capture_output=True, text=True)

        # Print the stdout and stderr from the script in real-time
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Check if the script executed successfully
        if result.returncode != 0:
            logging.error(f"--- Error in {script}. Pipeline stopped. ---")
            logging.error(f"Return Code: {result.returncode}")
            exit(1)  # Exit the script with a non-zero status code to indicate failure

        logging.info(f"--- Finished: {script} ---")

    logging.info("--- Data Analysis Pipeline Finished Successfully ---")

    # --- Convert .log file to .txt ---
    output_log_file_txt = LOG_FILE.replace('.log', '.txt')
    try:
        if os.path.exists(LOG_FILE):
            os.rename(LOG_FILE, output_log_file_txt)
            print(f"Log file successfully converted to: {output_log_file_txt}")
        else:
            print(f"Log file '{LOG_FILE}' not found to convert.")
    except Exception as e:
        print(f"Error converting log file: {e}")


if __name__ == '__main__':
    main()
