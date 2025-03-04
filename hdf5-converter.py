import os
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from joblib import Parallel, delayed  # For parallel processing
from app_config import get_config
from utils import set_up_logging
import logging
from pathlib import Path

# Load the models once (do not load inside the loop)
def get_model():
    """Load the Whisper feature extractor and tokenizer."""
    MODEL_TYPE = "openai/whisper-large-v3"
    TARGET_LANGUAGE = "german"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_TYPE)
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_TYPE, language=TARGET_LANGUAGE, task="transcribe")
    return feature_extractor, tokenizer

def process_audio_file(wav_path, transcript, float16=False, feature_extractor=None, tokenizer=None):
    """Process a single audio file and return the audio, features, and tokenized transcription."""

    if not os.path.isfile(wav_path):
        print(f"File does not exist: {wav_path}")
        return None, None, None

    # Load and resample the audio
    audio, _ = librosa.load(wav_path, sr=TARGET_SR)

    # Truncate or pad the audio to the desired length
    if len(audio) > MAX_LENGTH_SAMPLES:
        audio = audio[:MAX_LENGTH_SAMPLES]  # Truncate if longer than max_length_samples
    else:
        padded = np.zeros(MAX_LENGTH_SAMPLES, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    # Extract features from the audio using the Whisper feature extractor
    input_features = feature_extractor(audio, sampling_rate=TARGET_SR).input_features[0]

    # Tokenize the transcription using the Whisper tokenizer
    input_ids = tokenizer(str(transcript)).input_ids

    return (
        audio.astype("float32" if not float16 else "float16"),
        input_features.astype("float32" if not float16 else "float16"),
        np.array(input_ids, dtype=np.int32)
    )

def process_batch(batch_files, batch_transcripts, float16=False, feature_extractor=None, tokenizer=None):
    """Process a batch of audio files and return the audio, features, and tokenized transcriptions."""
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_audio_file)(wav_path, transcript, float16, feature_extractor, tokenizer)
        for wav_path, transcript in zip(batch_files, batch_transcripts)
    )

    # Filter out any results where the audio could not be processed
    valid_results = [res for res in results if res[0] is not None]
    if not valid_results:
        return None, None, None

    batch_audio_data, batch_features_data, batch_transcription_data = zip(*valid_results)

    return np.array(batch_audio_data), np.array(batch_features_data), list(batch_transcription_data)

def hdfconverter(BASE_DIR, CSV_FILE, OUTPUT_HDF5, feature_extractor, tokenizer, float16=False):
    """Convert a dataset to HDF5 format."""
    # Read the metadata CSV file
    df = pd.read_csv(CSV_FILE)
    file_names = df["file_name"].tolist()
    # Use an empty string if transcription is missing
    transcriptions = df.get("transcription", [""] * len(file_names))

    # Create the HDF5 file
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_file:

        # Determine the shape of input features using a dummy audio sample
        dummy_audio = np.zeros(MAX_LENGTH_SAMPLES, dtype=np.float32)
        dummy_features = feature_extractor(dummy_audio, sampling_rate=TARGET_SR).input_features[0]
        feature_shape = dummy_features.shape  # e.g., (128, 3000)

        # Create dataset for extracted input features (2D arrays per file)
        features_dataset = hdf5_file.create_dataset(
            "input_features",
            shape=(len(file_names), feature_shape[0], feature_shape[1]),
            dtype="float32" if not float16 else "float16",
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Create dataset for tokenized transcriptions (variable-length sequences of integers)
        dt_int = h5py.special_dtype(vlen=np.int32)
        transcript_dataset = hdf5_file.create_dataset(
            "transcription",
            shape=(len(file_names),),
            dtype=dt_int,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Process files in batches
        for i in tqdm(range(0, len(file_names), BATCH_SIZE), desc="Converting to HDF5", unit="batch"):
            # Build the full file paths for the current batch
            batch_files = [os.path.join(BASE_DIR, f) for f in file_names[i:i+BATCH_SIZE]]
            batch_transcripts = transcriptions[i:i+BATCH_SIZE]

            batch_audio, batch_features, batch_transcription = process_batch(
                batch_files, batch_transcripts, float16, feature_extractor, tokenizer
            )

            # Skip this batch if no valid files were processed
            if batch_audio is None:
                continue

            batch_len = len(batch_audio)
            # audio_dataset[i:i+batch_len] = batch_audio
            features_dataset[i:i+batch_len] = batch_features
            transcript_dataset[i:i+batch_len] = batch_transcription

    logging.info(f"Finished conversion to HDF5: {OUTPUT_HDF5.stem}")

def test_hdf5(hdf5_file):
    """Test the HDF5 file by printing its content."""
    logging.info(f"Testing HDF5 file: {hdf5_file.stem}")
    with h5py.File(hdf5_file, "r") as hdf5_file:
        print("Content of the HDF5 file:")
        for key in hdf5_file.keys():
            print(key)
        audio_data = hdf5_file["input_features"][:]
        transcriptions = hdf5_file["transcription"][:]

    print("Number of audio data:", len(audio_data))
    print("First audio data:", audio_data[0])
    print("First transcript", transcriptions[0])
    logging.info("HDF5 file test complete.")

def main():
    """Main function to execute the HDF5 conversion process."""
    # Set global variables
    global OUTPUT_DIRECTORY, TARGET_SR, BATCH_SIZE, N_JOBS, MODEL_TYPE, TARGET_LANGUAGE, MAX_LENGTH_SAMPLES, INPUT_DIRECTORY

    # Load configuration
    config = get_config()["hdf5_converter"]
    config_logging = get_config()["logging"]
    INPUT_DIRECTORY = Path(config["input_directory"])
    OUTPUT_DIRECTORY = Path(config["output_directory"])
    MAX_LENGTH_SAMPLES = int(config["max_length_samples"])
    TARGET_SR = int(config["sample_rate"])
    BATCH_SIZE = int(config["batch_size"])
    N_JOBS = int(config["threads"])
    MODEL_TYPE = config["model"]
    TARGET_LANGUAGE = config["language"]
    LOGGING_DIRECTORY = Path(config_logging["logging_directory"])

    # Set up logging
    logging_file_name = "hdf5_conversion_erros.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)
    logging.info("Starting hdf5 conversion.")

    # Load the Whisper feature extractor and tokenizer
    feature_extractor, tokenizer = get_model()

    for BASE_DIR in tqdm(list(INPUT_DIRECTORY.iterdir()), desc="Processing datasets", unit="dataset"):
        try:
            logging.info(f"Processing dataset: {BASE_DIR.name}")
            if BASE_DIR.is_dir():
                CSV_FILE = BASE_DIR / "metadata.csv"
                OUTPUT_HDF5 = OUTPUT_DIRECTORY / f"{BASE_DIR.stem}.hdf5"
                hdfconverter(BASE_DIR, CSV_FILE, OUTPUT_HDF5, feature_extractor, tokenizer, float16=False)
                test_hdf5(OUTPUT_HDF5)
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing dataset {BASE_DIR}: {e}")
            logger.removeHandler(error_file_handler)
            continue

    logging.info("HDF5 conversion complete.")

    
if __name__ == "__main__":
    main()