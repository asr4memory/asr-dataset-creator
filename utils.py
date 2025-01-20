import os
import logging

def list_files(directory):
    "Function to get the base name of the files up to the first underline."
    files = {}
    for filename in os.listdir(directory):
        base_name = filename.split("_", 1)[0]
        files[base_name] = filename
    return files

def set_up_logging(logging_directory, logging_file_name):
    "Set up logging."
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.captureWarnings(True)
    error_file_handler = logging.FileHandler(logging_directory / logging_file_name)
    error_file_handler.setLevel(logging.ERROR)  # Nur Errors in die Datei schreiben
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    error_file_handler.setFormatter(formatter)
    
    return error_file_handler