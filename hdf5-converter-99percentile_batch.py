import os
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from joblib import Parallel, delayed  # For parallel processing

# Set up paths and constants
BASE_DIR = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/dataset-merger/_output/eg_dataset_complete"
CSV_FILE = os.path.join(BASE_DIR, "metadata.csv")
OUTPUT_HDF5 = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/hdf-converter/_output/eg_dataset_complete_13sec_features_batch.h5"

TARGET_SR = 16000         # Target sample rate for audio
BATCH_SIZE = 400          # Batch size for processing
N_JOBS = 16                # Number of parallel workers (adjust based on your hardware)

# Load the models once (do not load inside the loop)
MODEL_TYPE = "openai/whisper-large-v3"
TARGET_LANGUAGE = "german"
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_TYPE)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_TYPE, language=TARGET_LANGUAGE, task="transcribe")

def process_audio_file(wav_path, transcript, max_length_samples, sr=TARGET_SR, float16=False):

    if not os.path.isfile(wav_path):
        print(f"File does not exist: {wav_path}")
        return None, None, None

    # Load and resample the audio
    audio, _ = librosa.load(wav_path, sr=sr)

    # Truncate or pad the audio to the desired length
    if len(audio) > max_length_samples:
        audio = audio[:max_length_samples]  # Truncate if longer than max_length_samples
    else:
        padded = np.zeros(max_length_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    # Extract features from the audio using the Whisper feature extractor
    input_features = feature_extractor(audio, sampling_rate=sr).input_features[0]

    # Tokenize the transcription using the Whisper tokenizer
    input_ids = tokenizer(str(transcript)).input_ids

    return (
        audio.astype("float32" if not float16 else "float16"),
        input_features.astype("float32" if not float16 else "float16"),
        np.array(input_ids, dtype=np.int32)
    )

def process_batch(batch_files, batch_transcripts, max_length_samples, sr=TARGET_SR, float16=False):
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_audio_file)(wav_path, transcript, max_length_samples, sr, float16)
        for wav_path, transcript in zip(batch_files, batch_transcripts)
    )

    # Filter out any results where the audio could not be processed
    valid_results = [res for res in results if res[0] is not None]
    if not valid_results:
        return None, None, None

    batch_audio_data, batch_features_data, batch_transcription_data = zip(*valid_results)

    return np.array(batch_audio_data), np.array(batch_features_data), list(batch_transcription_data)

def hdfconverter(max_length_samples, sr=TARGET_SR, float16=False):
    # Read the metadata CSV file
    df = pd.read_csv(CSV_FILE)
    file_names = df["file_name"].tolist()
    # Use an empty string if transcription is missing
    transcriptions = df.get("transcription", [""] * len(file_names))

    # Create the HDF5 file
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_file:
        # Create dataset for raw audio waveforms (1D arrays with length max_length_samples)
        # audio_dataset = hdf5_file.create_dataset(
        #     "audio_waveforms",
        #     shape=(len(file_names), max_length_samples),
        #     dtype="float32" if not float16 else "float16",
        #     compression="gzip",
        #     compression_opts=4,
        #     chunks=True
        # )

        # Determine the shape of input features using a dummy audio sample
        dummy_audio = np.zeros(max_length_samples, dtype=np.float32)
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
                batch_files, batch_transcripts, max_length_samples, sr, float16
            )

            # Skip this batch if no valid files were processed
            if batch_audio is None:
                continue

            batch_len = len(batch_audio)
            # audio_dataset[i:i+batch_len] = batch_audio
            features_dataset[i:i+batch_len] = batch_features
            transcript_dataset[i:i+batch_len] = batch_transcription

    print(f"Finished conversion to HDF5: {OUTPUT_HDF5}")

def main():
    hdfconverter(max_length_samples=208000, float16=False)

if __name__ == "__main__":
    main()