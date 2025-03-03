import os
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from joblib import Parallel, delayed  # Parallel processing

BASE_DIR = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/dataset-merger/_output/eg_dataset_complete"
CSV_FILE = os.path.join(BASE_DIR, "metadata.csv")
OUTPUT_HDF5 = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/hdf-converter/_output/eg_dataset_complete_13sec.h5"

TARGET_SR = 16000
BATCH_SIZE = 100  # Adjust batch size
N_JOBS = 4  # Number of parallel workers (adjust based on CPU/GPU)

# Load models once (NOT inside the loop)
MODEL_TYPE = "whisper-large-v3"
TARGET_LANGUAGE = "german"
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_TYPE)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_TYPE, language=TARGET_LANGUAGE, task="transcribe")


def process_audio_file(wav_path, transcript, max_length_samples, sr=TARGET_SR, float16=False):
    """Loads, processes, and extracts features for a single audio file"""
    if not os.path.isfile(wav_path):
        print(f"File does not exist: {wav_path}")
        return None, None

    # Load and resample audio
    audio, _ = librosa.load(wav_path, sr=sr)

    # Truncate or pad
    if len(audio) > max_length_samples:
        audio = audio[:max_length_samples]  # Truncate
    else:
        padded = np.zeros(max_length_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded

    # Feature extraction
    input_features = feature_extractor(audio, sampling_rate=sr).input_features[0]

    # Tokenize transcription
    input_ids = tokenizer(str(transcript)).input_ids

    return input_features.astype("float32" if not float16 else "float16"), np.array(input_ids, dtype=np.int32)


def process_batch(batch_files, batch_transcripts, max_length_samples, sr=TARGET_SR, float16=False):
    """Processes an entire batch of files in parallel"""
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_audio_file)(wav_path, transcript, max_length_samples, sr, float16)
        for wav_path, transcript in zip(batch_files, batch_transcripts)
    )

    # Separate valid results (ignore None results)
    batch_audio_data, batch_transcription_data = zip(*[res for res in results if res[0] is not None])

    return np.array(batch_audio_data), list(batch_transcription_data)


def hdfconverter(max_length_samples, sr=TARGET_SR, float16=False):
    """Converts dataset to HDF5 format with batch processing and parallelization"""
    df = pd.read_csv(CSV_FILE)
    file_names = df["file_name"].tolist()
    transcriptions = df.get("transcription", [""] * len(file_names))

    # Create HDF5 file
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_file:
        # Pre-allocate datasets
        audio_dataset = hdf5_file.create_dataset(
            "audio_waveforms",
            shape=(len(file_names), max_length_samples),
            dtype="float32" if not float16 else "float16",
            compression="gzip",
            compression_opts=4,
            chunks=True
        )
        transcript_dataset = hdf5_file.create_dataset(
            "transcription",
            shape=(len(file_names),),
            dtype=h5py.special_dtype(vlen=np.int32),  # Variable-length sequences
            compression="gzip",
            compression_opts=4,
            chunks=True
        )

        # Process in batches
        for i in tqdm(range(0, len(file_names), BATCH_SIZE), desc="Converting to HDF5", unit="batch"):
            batch_files = [os.path.join(BASE_DIR, f) for f in file_names[i:i+BATCH_SIZE]]
            batch_transcripts = transcriptions[i:i+BATCH_SIZE]

            # Process batch in parallel
            batch_audio, batch_transcription = process_batch(batch_files, batch_transcripts, max_length_samples, sr, float16)

            # Store in HDF5
            audio_dataset[i:i+len(batch_audio)] = batch_audio
            transcript_dataset[i:i+len(batch_transcription)] = batch_transcription

    print(f"Finished conversion to HDF5: {OUTPUT_HDF5}")


def main():
    hdfconverter(max_length_samples=208000, float16=False)


if __name__ == "__main__":
    main()
