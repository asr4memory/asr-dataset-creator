import os
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer

# Base paths and constants
BASE_DIR = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/dataset-merger/_output/eg_dataset_complete"
CSV_FILE = os.path.join(BASE_DIR, "metadata.csv")
OUTPUT_HDF5 = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/hdf-converter/_output/eg_dataset_complete_13sec_features_new.h5"

TARGET_SR = 16000
PERCENTILE = 99  # (Unused in this script)

# Load the models for preprocessing
model_type = "openai/whisper-large-v3"
target_language = "german"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")

def hdfconverter(max_length_samples, sr=TARGET_SR, float16=False):
    # Read the CSV file containing metadata
    df = pd.read_csv(CSV_FILE)
    file_names = df["file_name"].tolist()

    # Open HDF5 file for writing
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_file:
        # Create dataset for raw audio waveforms (each file is stored as a 1D array of length max_length_samples)
        # audio_dataset = hdf5_file.create_dataset(
        #     "audio_waveforms",
        #     shape=(len(file_names), max_length_samples),
        #     dtype="float32" if not float16 else "float16",
        #     compression="gzip",
        #     compression_opts=4,
        #     chunks=True
        # )

        # Determine the shape of the input features by processing a dummy audio sample
        dummy_audio = np.zeros(max_length_samples, dtype=np.float32)
        dummy_features = feature_extractor(dummy_audio, sampling_rate=sr).input_features[0]
        feature_shape = dummy_features.shape  # e.g., (128, 3000)

        # Create dataset for extracted input features (each file yields a 2D array)
        features_dataset = hdf5_file.create_dataset(
            "input_features",
            shape=(len(file_names), feature_shape[0], feature_shape[1]),
            dtype="float32" if not float16 else "float16",
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Create dataset for tokenized transcriptions (variable-length sequences of integers)
        dt_int = h5py.special_dtype(vlen=np.dtype('int32'))
        transcript_dataset = hdf5_file.create_dataset(
            "transcription",
            shape=(len(file_names),),
            dtype=dt_int,
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Iterate over each row in the CSV file
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Converting to HDF5", unit="file"):
            wav_file = row["file_name"]
            transcript = row.get("transcription", "")  # Use an empty string if transcription is missing

            # Build the full path to the audio file
            wav_path = os.path.join(BASE_DIR, wav_file)
            if not os.path.isfile(wav_path):
                print(f"File does not exist: {wav_path}")
                continue

            # Load the audio with librosa (resampled to TARGET_SR)
            audio, _ = librosa.load(wav_path, sr=sr)
            length_samples = len(audio)

            # Pad or truncate the audio to the fixed length
            if length_samples > max_length_samples:
                audio = audio[:max_length_samples]
            else:
                padded = np.zeros(max_length_samples, dtype=np.float32)
                padded[:length_samples] = audio
                audio = padded

            # Compute log-Mel input features using the feature extractor
            input_features = feature_extractor(audio, sampling_rate=sr).input_features[0]

            # Tokenize the transcription using the tokenizer
            input_ids = tokenizer(str(transcript)).input_ids

            # Save the raw audio, input features, and tokenized transcription to their respective datasets
            # audio_dataset[i] = audio
            features_dataset[i] = input_features
            transcript_dataset[i] = np.array(input_ids, dtype=np.int32)

    print(f"Finished conversion to HDF5: {OUTPUT_HDF5}")

def main():
    # You can compute max_length_samples using a percentile if needed; here, a fixed value is used.
    hdfconverter(max_length_samples=208000, float16=True)

if __name__ == "__main__":
    main()