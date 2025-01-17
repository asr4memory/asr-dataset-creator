import pandas as pd
import re
from pathlib import Path
from app_config import get_config
import os
import logging
from utils import list_files, set_up_logging

# Load the configuration
config = get_config()["vtt_anonymization"]

INPUT_DIR_VTT = Path(config["input_directory_vtt"])
INPUT_DIR_JSON = Path(config["input_directory_json"])
OUTPUT_DIR = Path(config["output_directory"])
LOGGING_DIRECTORY = Path(config["logging_directory"])

def load_json(file_path):
    """
    Loads the anonymized JSON file and processes it.
    """
    df = pd.read_json(file_path, encoding="utf-8")
    words_to_anonymize = df[df['is_historical'] == False]['entity_name'].tolist()
    return list(words_to_anonymize) 


def load_vtt(file_path):
    """
    Loads the VTT file content.
    """
    with open(file_path, "r", encoding="utf-8") as vtt_file:
        return vtt_file.read()


def apply_replacements(content, replacements):
    """
    Applies character replacements on the VTT content.
    """
    for item in replacements:
        pattern = item["pattern"]
        replacement = item["replacement"]
        content = re.sub(pattern, replacement, content)
    return content


def anonymize_words(content, words_to_anonymize):
    """
    Replaces specified words in the VTT content.
    """
    for word in words_to_anonymize:
        word_regex = r'(?:.*\n){0,3}.*\b' + re.escape(word) + r'\b.*\n?'
        content = re.sub(word_regex, '', content)
    return content


def save_vtt(content, output_path):
    """
    Saves the modified VTT content to a file.
    """
    with open(output_path, "w", encoding="utf-8") as vtt_file:
        vtt_file.write(content)
    print(f"Anonymized VTT file saved at: {output_path}")


def main():
    """Main function to execute the VTT anonymization process."""
    # Set up logging
    error_file_handler = set_up_logging(LOGGING_DIRECTORY)
    logging.info("Starting VTT anonymization workflow.")

    # List all VTT and JSON files in the input directories
    input_vtt_files = list_files(INPUT_DIR_VTT)
    input_json_files = list_files(INPUT_DIR_JSON)

    for vtt_base, vtt_filename in input_vtt_files.items():
        try:
            logging.info(f"Processing file: {vtt_filename}")

            if vtt_base not in input_json_files:
                continue

            if vtt_filename == ".DS_Store":
                continue

            # Construct complete path to the files:
            input_vtt_file = INPUT_DIR_VTT / vtt_filename
            input_json_file = INPUT_DIR_JSON / input_json_files[vtt_base]   

            # Load JSON and extract words to anonymize
            words_to_anonymize = load_json(input_json_file)
            logging.info(f"Words to anonymize: {words_to_anonymize}")

            # Load VTT file content
            vtt_content = load_vtt(input_vtt_file)

            # Apply replacements
            replacements = config.get("vtt_replacements", [])
            vtt_content = apply_replacements(vtt_content, replacements)

            # Anonymize words
            vtt_content = anonymize_words(vtt_content, words_to_anonymize)

            # Save the modified VTT content
            output_vtt_file = OUTPUT_DIR / f"{input_vtt_file.stem}_anonymized.vtt"
            save_vtt(vtt_content, output_vtt_file)
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing file {vtt_filename}: {e}")
            logger.removeHandler(error_file_handler)
            continue


if __name__ == "__main__":
    main()