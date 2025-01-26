from pathlib import Path
import re
import string
from app_config import get_config
import logging
from utils import set_up_logging

def clean_vtt_content(content, remove_punctuation=False):
    "Cleans up VTT strings so that WER can be detected."
    # Remove the VTT heading, segment numbers, time codes and notes and
    # comments in () and <>:
    result = re.sub(
        r"WEBVTT\n\n|\d+\n\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n|\(.*?\)|<.*?>",
        "",
        content,
    )
    # Corrected regular expression to remove the filler words "äh",
    # "ähs", "ähm", "hm", und "hmm" including the following comma,
    # semicolon, hyphen or period:
    result = re.sub(
        r"\b(äh|ähs|ähm|hm|hmm)\b\s*[,;:\-\.]?\s*", "", result, flags=re.IGNORECASE
    )
    # Remove underlines:
    result = re.sub(r"_", "", result)
    # Removing quotation marks:
    result = re.sub(r'[\'"]', "", result)
    # Remove all forms of blank lines:
    result = re.sub(r"^\s*$\n", "", result, flags=re.MULTILINE)
    # Removing all new lines
    # result = re.sub(r'\n', ' ', result)

    # Additional removal of all punctuation if requested:
    if remove_punctuation:
        # Remove all punctuation except . and : after numbers:
        punctuation_to_remove = string.punctuation.replace(".", "").replace(":", "")
        result = re.sub(
            r"(?<!\d)[{}]+".format(re.escape(punctuation_to_remove)), "", result
        )
        # Additional removal of punctuation that does not
        # follow numbers:
        result = re.sub(r"(?<=\D)[.:]+", "", result)

    return result

def run_on_directory():
    # Get the configuration settings
    config = get_config()["vtt_to_txt"]
    config_logging = get_config()["logging"]
    INPUT_PATH = Path(config["input_directory"])
    OUTPUT_PATH = Path(config["output_directory"])
    LOGGING_DIRECTORY = Path(config_logging["logging_directory"])
    "Run through all VTT files in the specified directory."   
     
    # Set up logging
    logging_file_name = "vtt_to_txt_errors.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)

    logging.info("Starting VTT to TXT conversion.")
    for filepath in INPUT_PATH.glob("*.vtt"):
        try:
            logging.info(f"Processing file: {filepath}")
            orig_stem = filepath.stem
            vtt_content = filepath.read_text(encoding="utf-8")

            # Write cleaned up output file (first version).
            cleaned_text = clean_vtt_content(vtt_content)
            new_filepath = OUTPUT_PATH / f"{orig_stem}.cleared.txt"
            new_filepath.write_text(cleaned_text, encoding="utf-8")
            logging.info(f"Cleaned up file written to: {new_filepath}")
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing file {filepath}: {e}")
            logger.removeHandler(error_file_handler)
            continue

if __name__ == "__main__":
    run_on_directory()