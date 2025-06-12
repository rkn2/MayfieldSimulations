import logging
import sys

def setup_logging(log_file='pipeline.log'):
    """
    Sets up logging to both a file and the console.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the lowest level of messages to capture

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Add both handlers to the root logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)