import csv
from pathlib import Path
import shutil
from tqdm import tqdm
from app_config import get_config
import logging
from utils import set_up_logging

def setup_output_structure(OUTPUT_FOLDER, OUTPUT_DATA_FOLDER):
    """Create the output directory and data subfolder if they do not exist."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_FOLDER.mkdir(parents=True, exist_ok=True)


def initialize_metadata_file(OUTPUT_METADATA_FILE):
    """Initialize the combined metadata.csv file with a header."""
    with OUTPUT_METADATA_FILE.open(mode='w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["file_name", "transcription"])


def process_dataset(dataset_path, OUTPUT_DATA_FOLDER, MERGED_DATASET_FOLDER, OUTPUT_METADATA_FILE):
    """Process a single dataset: copy audio files and update the combined metadata."""
    metadata_file = dataset_path / 'metadata.csv'
    if not metadata_file.exists():
        return  # Skip if metadata.csv does not exist in this dataset

    with metadata_file.open(mode='r', encoding='utf-8') as metafile:
        csvreader = csv.reader(metafile)
        next(csvreader, None)  # Skip the header

        for row in tqdm(csvreader, desc=f"Copying files from {dataset_path.name}", unit="file"):
            segment_path = dataset_path / row[0]
            if segment_path.exists():
                new_segment_name = f"{segment_path.name}"
                new_segment_path = OUTPUT_DATA_FOLDER / new_segment_name

                # Copy the segment to the unified data folder
                shutil.copy(segment_path, new_segment_path)

                # Append entry to the combined metadata.csv
                append_to_metadata(new_segment_path.relative_to(MERGED_DATASET_FOLDER), row[1], OUTPUT_METADATA_FILE)


def append_to_metadata(file_name, transcription, OUTPUT_METADATA_FILE):
    """Append a single entry to the combined metadata file."""
    with OUTPUT_METADATA_FILE.open(mode='a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([file_name, transcription])


def merge_datasets():
    """Main function for the merging of datasets."""
    # Load the configuration
    config = get_config()["dataset_merger"]
    config_logging = get_config()["logging"]

    DATASETS_FOLDER = Path(config["input_directory"])
    OUTPUT_FOLDER = Path(config["output_directory"])
    MERGED_DATASET_NAME = config["merged_dataset_name"]
    MERGED_DATASET_FOLDER = OUTPUT_FOLDER / MERGED_DATASET_NAME
    OUTPUT_DATA_FOLDER = MERGED_DATASET_FOLDER/ 'data'
    OUTPUT_METADATA_FILE = MERGED_DATASET_FOLDER/ 'metadata.csv'
    LOGGING_DIRECTORY = Path(config_logging["logging_directory"])

    # Set up logging
    logging_file_name = "merge_datasets_erros.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)
    logging.info("Starting dataset merging.")
    
    setup_output_structure(OUTPUT_DATA_FOLDER, OUTPUT_DATA_FOLDER)
    initialize_metadata_file(OUTPUT_METADATA_FILE)

    for dataset_path in DATASETS_FOLDER.iterdir():
        try:
            logging.info(f"Processing dataset: {dataset_path.name}")
            if dataset_path.is_dir():
                process_dataset(dataset_path, OUTPUT_DATA_FOLDER, MERGED_DATASET_FOLDER, OUTPUT_METADATA_FILE)
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing dataset {dataset_path}: {e}")
            logger.removeHandler(error_file_handler)
            continue

    logging.info("Dataset merging completed.")


if __name__ == "__main__":
    merge_datasets()