import os
import h5py
import librosa
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel processing
from app_config import get_config
from utils import set_up_logging
import logging
from pathlib import Path

def process_audio_file(wav_path, transcript, target_sr=16000, max_length_seconds=30):
    """Loads an audio file, resamples to target sample rate, and pads/truncates to fixed length."""

    if not os.path.isfile(wav_path):
        print(f"File does not exist: {wav_path}")
        return None, None, None
    
    try:
        # Load the audio file with the target sample rate
        audio, sample_rate = librosa.load(wav_path, sr=target_sr)
        
        # Calculate max length in samples
        max_length_samples = target_sr * max_length_seconds
        
        # Pad or truncate the audio to the fixed length
        if len(audio) > max_length_samples:
            # Truncate if longer than max_length_samples
            audio = audio[:max_length_samples]
        else:
            # Pad with zeros if shorter
            padded = np.zeros(max_length_samples, dtype=np.float32)
            padded[:len(audio)] = audio
            audio = padded
            
        # Return the padded/truncated audio array and the transcript
        return audio.astype(np.float32), str(transcript), target_sr
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None, None, None

def process_batch(batch_files, batch_transcripts, target_sr=16000, max_length_seconds=30, n_jobs=4):
    """Processes a batch of audio files with fixed length padding/truncation."""
    # Calculate max length in samples for pre-allocation
    max_length_samples = target_sr * max_length_seconds
    
    results = Parallel(n_jobs=n_jobs, verbose=10, prefer="threads")(
        delayed(process_audio_file)(wav_path, transcript, target_sr, max_length_seconds)
        for wav_path, transcript in zip(batch_files, batch_transcripts)
    )

    # Filter out any results where the audio could not be processed
    valid_results = [res for res in results if res[0] is not None]
    if not valid_results:
        return None, None, None

    batch_audio_data, batch_transcriptions, sample_rates = zip(*valid_results)

    return list(batch_audio_data), list(batch_transcriptions), list(sample_rates)

def hdfconverter(BASE_DIR, CSV_FILE, OUTPUT_HDF5, target_sr=16000, max_length_seconds=30, n_jobs=4, batch_size=100):
    """Converts a dataset to HDF5 format with fixed-length audio arrays using optimized writing."""
    logging.info(f"Converting {CSV_FILE} with {n_jobs} parallel jobs")
    
    # Read the metadata CSV file
    df = pd.read_csv(CSV_FILE)
    file_names = df["file_name"].tolist()
    # Use an empty string if transcription is missing
    transcriptions = df.get("transcription", [""] * len(file_names))

    # Calculate max length in samples
    max_length_samples = target_sr * max_length_seconds
    
    # Calculate optimal chunk size for HDF5
    # For audio data, use chunk size that matches a small number of complete audio samples
    chunk_samples = min(max_length_samples, 16384)  # 16K samples per chunk or less
    audio_chunks = (100, chunk_samples)  # Optimize for batch writes of 100 files
    
    # For string data, make smaller chunks
    str_chunks = 100  # 100 strings per chunk

    # Create the HDF5 file with optimized settings
    with h5py.File(OUTPUT_HDF5, "w", libver='latest') as hdf5_file:
        # Create dataset for audio data with fixed length and optimized parameters
        audio_dataset = hdf5_file.create_dataset(
            "audio",
            shape=(len(file_names), max_length_samples),
            dtype=np.float32,
            compression="lzf",  # Use faster LZF compression instead of gzip
            chunks=audio_chunks,
            shuffle=True,  # Improves compression
            fletcher32=False  # Disable checksums for better performance
        )

        # Create dataset for transcriptions (strings) with optimized parameters
        dt_str = h5py.special_dtype(vlen=str)
        transcript_dataset = hdf5_file.create_dataset(
            "transcription",
            shape=(len(file_names),),
            dtype=dt_str,
            compression="lzf",
            chunks=(str_chunks,),
            fletcher32=False
        )
        
        # Create dataset for sample rates
        sample_rate_dataset = hdf5_file.create_dataset(
            "sample_rate",
            shape=(len(file_names),),
            dtype=np.int32,
            compression="lzf",
            chunks=(str_chunks,),
            fletcher32=False
        )

        # Create dataset for filenames
        filename_dataset = hdf5_file.create_dataset(
            "filename",
            shape=(len(file_names),),
            dtype=dt_str,
            compression="lzf",
            chunks=(str_chunks,),
            fletcher32=False
        )
        
        # Store the filenames in one batch operation
        for i in range(0, len(file_names), str_chunks):
            end_idx = min(i + str_chunks, len(file_names))
            batch_filenames = file_names[i:end_idx]
            filename_dataset[i:end_idx] = batch_filenames

        # Process files in batches
        for i in tqdm(range(0, len(file_names), batch_size), desc="Converting to HDF5", unit="batch"):
            batch_start_time = time.time()
            
            batch_end = min(i + batch_size, len(file_names))
            batch_files = [os.path.join(BASE_DIR, f) for f in file_names[i:batch_end]]
            batch_transcripts = transcriptions[i:batch_end]

            batch_audio, batch_transcription, batch_sample_rates = process_batch(
                batch_files, batch_transcripts, target_sr, max_length_seconds, n_jobs
            )

            # Skip this batch if no valid files were processed
            if batch_audio is None:
                continue

            batch_len = len(batch_audio)
            
            # Create numpy arrays for batch storage
            audio_batch_array = np.array(batch_audio, dtype=np.float32)
            
            # Measure HDF5 write time
            hdf5_write_start = time.time()
            
            # Write all data at once using array slicing - much faster than individual writes
            audio_dataset[i:i+batch_len] = audio_batch_array
            transcript_dataset[i:i+batch_len] = batch_transcription
            sample_rate_dataset[i:i+batch_len] = batch_sample_rates
            
            hdf5_write_time = time.time() - hdf5_write_start
            batch_total_time = time.time() - batch_start_time
            
            # Log detailed timing information for performance analysis
            if i % 5 == 0:
                logging.info(f"Batch {i//batch_size+1}/{len(file_names)//batch_size+1}: "
                             f"Audio processing: {batch_total_time-hdf5_write_time:.2f}s, "
                             f"HDF5 write: {hdf5_write_time:.2f}s, "
                             f"Total: {batch_total_time:.2f}s")
            
            # Call flush occasionally to ensure data is written to disk
            if i % (batch_size * 10) == 0:
                hdf5_file.flush()

    logging.info(f"Conversion to HDF5 completed: {OUTPUT_HDF5.stem}")

def test_hdf5(hdf5_file):
    """Tests the HDF5 file by printing its content."""
    logging.info(f"Testing HDF5 file: {hdf5_file.stem}")
    with h5py.File(hdf5_file, "r") as hdf5_file:
        print("Content of the HDF5 file:")
        for key in hdf5_file.keys():
            print(f"- {key}: {hdf5_file[key].shape}")
        
        # Display sample data
        if len(hdf5_file["audio"]) > 0:
            first_audio = hdf5_file["audio"][0]
            first_transcript = hdf5_file["transcription"][0]
            first_sample_rate = hdf5_file["sample_rate"][0]
            first_filename = hdf5_file["filename"][0]
            
            # Calculate non-zero samples (actual audio length before padding)
            non_zero_samples = np.sum(first_audio != 0)
            audio_duration_seconds = non_zero_samples / first_sample_rate
            
            print(f"\nExample (1st entry):")
            print(f"Filename: {first_filename}")
            print(f"Audio: {len(first_audio)} samples ({len(first_audio)/first_sample_rate:.2f} seconds), Original content: {audio_duration_seconds:.2f} seconds")
            print(f"Sample Rate: {first_sample_rate} Hz")
            print(f"Transcript: {first_transcript}")
            print(f"Audio stats: min={first_audio.min():.4f}, max={first_audio.max():.4f}, mean={first_audio.mean():.4f}")

    logging.info("HDF5 file test completed.")

def main():
    """Main function to execute the HDF5 conversion process."""
    # Load configuration
    config = get_config()["hdf5_converter"]
    config_logging = get_config()["logging"]
    INPUT_DIRECTORY = Path(config["input_directory"])
    OUTPUT_DIRECTORY = Path(config["output_directory"])
    BATCH_SIZE = int(config["batch_size"])
    N_JOBS = int(config["threads"])
    LOGGING_DIRECTORY = Path(config_logging["logging_directory"])
    
    # Set Whisper-specific parameters
    TARGET_SR = 16000  # Whisper uses 16kHz audio
    MAX_LENGTH_SECONDS = 30  # Standard 30-second clips for Whisper
    
    # For large datasets, optimize batch size for I/O performance
    if BATCH_SIZE > 200:
        logging.info(f"Reducing batch size from {BATCH_SIZE} to 200 for better I/O performance")
        BATCH_SIZE = 200

    # Ensure output directory exists
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging_file_name = "hdf5_conversion_errors.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)
    logging.info("Starting HDF5 conversion.")
    logging.info(f"Using {N_JOBS} parallel jobs for processing")
    logging.info(f"Batch size: {BATCH_SIZE} files per batch (consider reducing to 50-100 for better parallelization)")
    logging.info(f"Target sample rate: {TARGET_SR} Hz, Max audio length: {MAX_LENGTH_SECONDS} seconds")

    for BASE_DIR in tqdm(list(INPUT_DIRECTORY.iterdir()), desc="Processing datasets", unit="dataset"):
        try:
            logging.info(f"Processing dataset: {BASE_DIR.name}")
            if BASE_DIR.is_dir():
                CSV_FILE = BASE_DIR / "metadata.csv"
                OUTPUT_HDF5 = OUTPUT_DIRECTORY / f"{BASE_DIR.stem}.h5"
                
                start_time = time.time()
                hdfconverter(BASE_DIR, CSV_FILE, OUTPUT_HDF5, TARGET_SR, MAX_LENGTH_SECONDS, N_JOBS, BATCH_SIZE)
                conversion_time = time.time() - start_time
                logging.info(f"Dataset conversion completed in {conversion_time:.2f} seconds")
                
                test_hdf5(OUTPUT_HDF5)
                
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing dataset {BASE_DIR}: {e}")
            logger.removeHandler(error_file_handler)
            continue

    logging.info("HDF5 conversion completed.")

    
if __name__ == "__main__":
    main()