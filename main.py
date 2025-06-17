import os
import subprocess
import logging
import config


def main():
    """Orchestrates the execution of the entire data analysis pipeline."""
    if os.path.exists(config.PIPELINE_LOG_PATH):
        os.remove(config.PIPELINE_LOG_PATH)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.PIPELINE_LOG_PATH),
            logging.StreamHandler()
        ]
    )

    logging.info("--- Starting the Data Analysis Pipeline ---")

    scripts_to_run = [
        '1_dataCleaning.py',
        '2_dataPreprocessing.py',
        'edaSimDif.py',
        'featureImportance.py',
        '4_modeling.py',
        'shap_interpretation.py',
        #'6_deltaAccuracy.py', # not meaningful bc low accuracy models
        #'7_shap.py', # not meaningful bc low accuracy models
        '8_generate_report.py'
    ]

    for script in scripts_to_run:
        logging.info(f"--- Running: {script} ---")
        result = subprocess.run(['python', script], capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            logging.error(f"--- Error in {script}. Pipeline stopped. ---")
            exit(1)
        logging.info(f"--- Finished: {script} ---")

    logging.info("--- Data Analysis Pipeline Finished Successfully ---")


if __name__ == '__main__':
    main()
