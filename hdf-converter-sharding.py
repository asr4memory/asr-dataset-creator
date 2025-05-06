import os
import h5py
import librosa
import numpy as np
import pandas as pd
import time
import json
import random
from tqdm import tqdm
from joblib import Parallel, delayed  # For parallel processing
from app_config import get_config
from utils import set_up_logging
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split

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

def create_hdf5_datasets(hdf5_file, num_samples, max_length_samples, chunk_samples, str_chunks):
    """Helper function to create datasets in an HDF5 file with proper chunking."""
    # Create audio dataset with optimal chunking
    audio_chunks = (100, chunk_samples)
    audio_dataset = hdf5_file.create_dataset(
        "audio",
        shape=(num_samples, max_length_samples),
        dtype=np.float32,
        compression="gzip",
        compression_opts=9,
        chunks=audio_chunks,
        shuffle=True,
        fletcher32=False
    )
    
    # Create string datasets with optimal chunking
    dt_str = h5py.special_dtype(vlen=str)
    transcript_dataset = hdf5_file.create_dataset(
        "transcription",
        shape=(num_samples,),
        dtype=dt_str,
        compression="gzip",
        compression_opts=9,
        chunks=(str_chunks,),
        fletcher32=False
    )
    
    sample_rate_dataset = hdf5_file.create_dataset(
        "sample_rate",
        shape=(num_samples,),
        dtype=np.int32,
        compression="gzip",
        compression_opts=9,
        chunks=(str_chunks,),
        fletcher32=False
    )
    
    filename_dataset = hdf5_file.create_dataset(
        "filename",
        shape=(num_samples,),
        dtype=dt_str,
        compression="gzip",
        compression_opts=9,
        chunks=(str_chunks,),
        fletcher32=False
    )     
    
    return audio_dataset, transcript_dataset, sample_rate_dataset, filename_dataset

def split_dataset(csv_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=1337):
    """
    Split the dataset into training, validation, and test sets based on the given ratios.
    Returns DataFrames for each split.
    """
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    logging.info(f"Loaded dataset with {len(df)} entries from {csv_file}")
    
    # First split: training vs rest
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio, 
        random_state=random_seed
    )
    
    # Calculate the ratio for the validation set from the remaining data
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    
    # Second split: validation vs test from the remaining data
    val_df, test_df = train_test_split(
        temp_df, 
        train_size=val_size_adjusted, 
        random_state=random_seed
    )
    
    # Log split sizes
    logging.info(f"Split dataset: Training: {len(train_df)} ({len(train_df)/len(df):.1%}), "
                 f"Validation: {len(val_df)} ({len(val_df)/len(df):.1%}), "
                 f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df

def save_split_metadata(output_dir, dataset_name, train_df, val_df, test_df):
    """Save split metadata to CSV files in the respective directories."""
    # Create directories for each split
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    return train_dir, val_dir, test_dir


def hdfconverter_sharded(BASE_DIR, df, OUTPUT_DIR, split_name, dataset_name, shard_size=1000, target_sr=16000, 
                       max_length_seconds=30, n_jobs=4, batch_size=100):
    """Converts a dataset to multiple HDF5 shards with fixed-length audio arrays."""
    logging.info(f"Converting {split_name} split with {n_jobs} parallel jobs using sharding")
    
    # Extract information from the dataframe
    file_names = df["file_name"].tolist()
    # Use an empty string if transcription is missing
    transcriptions = df.get("transcription", [""] * len(file_names)).tolist()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Calculate max length in samples
    max_length_samples = target_sr * max_length_seconds
    
    # Calculate optimal chunk size for HDF5
    chunk_samples = min(max_length_samples, 16384)  # 16K samples per chunk or less
    str_chunks = 100  # 100 strings per chunk
    
    # For metadata tracking
    shard_info = {
        "dataset_name": dataset_name,
        "split_name": split_name,
        "total_files": len(file_names),
        "shard_size": shard_size,
        "num_shards": (len(file_names) + shard_size - 1) // shard_size,
        "shards": []
    }
    
    # Process files in shards
    for shard_idx in range(0, len(file_names), shard_size):
        shard_end = min(shard_idx + shard_size, len(file_names))
        shard_file_names = file_names[shard_idx:shard_end]
        shard_transcriptions = transcriptions[shard_idx:shard_end]
        
        # Create a unique filename for this shard WITHOUT REDUNDANCY
        shard_name = f"{dataset_name}_{split_name}_shard_{shard_idx//shard_size:04d}.h5"
        shard_path = os.path.join(OUTPUT_DIR, shard_name)
        
        current_shard = {
            "shard_id": shard_idx//shard_size,
            "shard_name": shard_name,
            "num_files": len(shard_file_names),
            "samples": []
        }
        
        logging.info(f"Creating {split_name} shard {shard_idx//shard_size + 1}/{(len(file_names)+shard_size-1)//shard_size} "
                    f"with {len(shard_file_names)} files: {shard_name}")
        
        # Create the HDF5 file for this shard
        with h5py.File(shard_path, "w", libver='latest') as hdf5_file:
            # Store shard metadata
            hdf5_file.attrs["shard_id"] = shard_idx//shard_size
            hdf5_file.attrs["dataset_name"] = dataset_name
            hdf5_file.attrs["split_name"] = split_name
            hdf5_file.attrs["total_shards"] = (len(file_names) + shard_size - 1) // shard_size
            hdf5_file.attrs["shard_start_idx"] = shard_idx
            hdf5_file.attrs["shard_end_idx"] = shard_end
            
            # Create datasets with optimal chunking
            audio_dataset, transcript_dataset, sample_rate_dataset, filename_dataset = create_hdf5_datasets(
                hdf5_file, len(shard_file_names), max_length_samples, chunk_samples, str_chunks
            )
            
            # Store the filenames for this shard in one operation
            filename_dataset[:] = shard_file_names
            
            # Process files in batches within this shard
            for batch_idx in tqdm(range(0, len(shard_file_names), batch_size), 
                                desc=f"Converting {split_name} shard {shard_idx//shard_size + 1}", unit="batch"):
                batch_start_time = time.time()
                
                # Calculate batch indices relative to the shard
                batch_end = min(batch_idx + batch_size, len(shard_file_names))
                batch_files = [os.path.join(BASE_DIR, f) for f in shard_file_names[batch_idx:batch_end]]
                batch_transcripts = shard_transcriptions[batch_idx:batch_end]
                
                # Process this batch of files
                batch_audio, batch_transcription, batch_sample_rates = process_batch(
                    batch_files, batch_transcripts, target_sr, max_length_seconds, n_jobs
                )
                
                # Skip this batch if no valid files were processed
                if batch_audio is None:
                    continue
                
                batch_len = len(batch_audio)
                
                # Create numpy arrays for batch storage
                audio_batch_array = np.array(batch_audio, dtype=np.float32)
                
                # Collect detailed sample information if requested
                for i in range(batch_len):
                    # Get the actual file index in this shard
                    sample_idx = batch_idx + i
                    
                    # Skip if we're somehow out of bounds
                    if sample_idx >= len(shard_file_names):
                        continue
                        
                    # Get sample data
                    audio = audio_batch_array[i]
                    filename = shard_file_names[sample_idx]
                    transcript = batch_transcription[i]
                    sample_rate = batch_sample_rates[i]
                    
                    # Calculate stats
                    non_zero_samples = np.sum(audio != 0)
                    audio_duration_seconds = non_zero_samples / sample_rate
                    audio_min = float(np.min(audio))
                    audio_max = float(np.max(audio))
                    audio_mean = float(np.mean(audio))
                    
                    # Store sample info
                    sample_info = {
                        "index": sample_idx,
                        "global_index": shard_idx + sample_idx,
                        "filename": filename,
                        "transcript": transcript,
                        "sample_rate": int(sample_rate),
                        "total_samples": int(len(audio)),
                        "non_zero_samples": int(non_zero_samples),
                        "duration_seconds": float(audio_duration_seconds),
                        "audio_stats": {
                            "min": audio_min,
                            "max": audio_max,
                            "mean": audio_mean
                        }
                    }
                    current_shard["samples"].append(sample_info)
                
                # Measure HDF5 write time
                hdf5_write_start = time.time()
                
                # Write batch data to HDF5 (using shard-relative indices)
                audio_dataset[batch_idx:batch_idx+batch_len] = audio_batch_array
                transcript_dataset[batch_idx:batch_idx+batch_len] = batch_transcription
                sample_rate_dataset[batch_idx:batch_idx+batch_len] = batch_sample_rates
                
                hdf5_write_time = time.time() - hdf5_write_start
                batch_total_time = time.time() - batch_start_time
                
                # Log timing information occasionally
                if batch_idx % (batch_size * 2) == 0:
                    logging.info(f"{split_name.capitalize()} Shard {shard_idx//shard_size + 1}, Batch {batch_idx//batch_size+1}: "
                                f"Processing: {batch_total_time-hdf5_write_time:.2f}s, "
                                f"Write: {hdf5_write_time:.2f}s")
                
            # Flush at the end of each shard
            hdf5_file.flush()
        
        # Add this shard's info to the overall info
        shard_info["shards"].append(current_shard)
    
    # Save shard metadata to JSON for easier access
    shard_metadata_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{split_name}_shards.json")
    with open(shard_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(shard_info, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Conversion of {split_name} split to sharded HDF5 completed: {len(shard_info['shards'])} shards created")
    return shard_info, OUTPUT_DIR  # Return shard info and output directory

def test_sharded_hdf5(shard_info, shard_dir):
    """Tests the sharded HDF5 files by sampling from each shard."""
    logging.info(f"Testing sharded HDF5 for {shard_info['split_name']} split of {shard_info['dataset_name']}")
    
    # Test first shard in detail, others just check existence
    if shard_info["shards"]:
        first_shard = shard_info["shards"][0]
        shard_path = os.path.join(shard_dir, first_shard["shard_name"])
        
        with h5py.File(shard_path, "r") as hdf5_file:
            logging.info(f"Checking shard 1/{shard_info['num_shards']}: {first_shard['shard_name']}")
            
            # Print shard metadata
            print(f"\n{shard_info['split_name'].capitalize()} shard metadata:")
            for key, value in hdf5_file.attrs.items():
                print(f"- {key}: {value}")
            
            # Print dataset information
            print("\nDatasets in this shard:")
            for key in hdf5_file.keys():
                print(f"- {key}: {hdf5_file[key].shape}")
            
            # Sample first entry in this shard
            if len(hdf5_file["audio"]) > 0:
                first_audio = hdf5_file["audio"][0]
                first_transcript = hdf5_file["transcription"][0]
                first_sample_rate = hdf5_file["sample_rate"][0]
                first_filename = hdf5_file["filename"][0]
                
                # Calculate non-zero samples (actual audio length before padding)
                non_zero_samples = np.sum(first_audio != 0)
                audio_duration_seconds = non_zero_samples / first_sample_rate
                
                print(f"\nSample from {shard_info['split_name']} shard (1st entry):")
                print(f"Filename: {first_filename}")
                print(f"Audio: {len(first_audio)} samples, Original content: {audio_duration_seconds:.2f} seconds")
                print(f"Transcript: {first_transcript}")
                print(f"Audio stats: min={first_audio.min():.4f}, max={first_audio.max():.4f}")
        
        # Check if all other shards exist
        for shard in shard_info["shards"][1:]:
            shard_path = os.path.join(shard_dir, shard["shard_name"])
            if not os.path.exists(shard_path):
                logging.warning(f"Shard {shard['shard_id']+1} at {shard['shard_name']} is missing")
    
    logging.info(f"{shard_info['split_name'].capitalize()} sharded HDF5 test completed")

def process_dataset_with_splits(BASE_DIR, CSV_FILE, OUTPUT_DIR, shard_size=1000, 
                              target_sr=16000, max_length_seconds=30, n_jobs=4, batch_size=100,
                              train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=1337):
    """Process a dataset by splitting it and creating sharded HDF5 files for each split."""
    # Verwende den tatsächlichen Datensatznamen
    dataset_name = os.path.basename(BASE_DIR)
    logging.info(f"Processing dataset {dataset_name} with train/val/test splits")
    
    # Create the main output directory
    dataset_output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # Split the dataset
    train_df, val_df, test_df = split_dataset(
        CSV_FILE, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Save split metadata and get split directories
    train_dir, val_dir, test_dir = save_split_metadata(
        dataset_output_dir, dataset_name, train_df, val_df, test_df
    )
    
    # Process each split
    split_info = {}
    
    # Process training split
    start_time = time.time()
    train_shard_info, train_shard_dir = hdfconverter_sharded(
        BASE_DIR, train_df, train_dir, "train", dataset_name,
        shard_size, target_sr, max_length_seconds, n_jobs, batch_size
    )
    logging.info(f"Training split conversion completed in {time.time() - start_time:.2f} seconds")
    split_info["train"] = train_shard_info
    
    # Process validation split
    start_time = time.time()
    val_shard_info, val_shard_dir = hdfconverter_sharded(
        BASE_DIR, val_df, val_dir, "val", dataset_name,
        shard_size, target_sr, max_length_seconds, n_jobs, batch_size
    )
    logging.info(f"Validation split conversion completed in {time.time() - start_time:.2f} seconds")
    split_info["val"] = val_shard_info
    
    # Process test split
    start_time = time.time()
    test_shard_info, test_shard_dir = hdfconverter_sharded(
        BASE_DIR, test_df, test_dir, "test", dataset_name,
        shard_size, target_sr, max_length_seconds, n_jobs, batch_size
    )
    logging.info(f"Test split conversion completed in {time.time() - start_time:.2f} seconds")
    split_info["test"] = test_shard_info

    # Test the sharded HDF5 files for each split
    test_sharded_hdf5(train_shard_info, train_shard_dir)
    test_sharded_hdf5(val_shard_info, val_shard_dir)
    test_sharded_hdf5(test_shard_info, test_shard_dir)
    
    # Save all split info to a single metadata file, but without sample details
    summary_split_info = {}
    for split_name, info in split_info.items():
        summary_split_info[split_name] = {
            "dataset_name": info["dataset_name"],
            "split_name": info["split_name"],
            "total_files": info["total_files"],
            "num_shards": info["num_shards"]
        }
    
    all_splits_info = {
        "dataset_name": dataset_name,
        "total_files": len(train_df) + len(val_df) + len(test_df),
        "train_files": len(train_df),
        "val_files": len(val_df),
        "test_files": len(test_df),
        "train_ratio": len(train_df) / (len(train_df) + len(val_df) + len(test_df)),
        "val_ratio": len(val_df) / (len(train_df) + len(val_df) + len(test_df)),
        "test_ratio": len(test_df) / (len(train_df) + len(val_df) + len(test_df)),
        "ramdiom_seed": random_seed,
        "splits": summary_split_info
    }
    # Save all split info to JSON with ensure_ascii=False for better readability
    with open(os.path.join(dataset_output_dir, f"{dataset_name}_split_info.json"), 'w', encoding='utf-8') as f:
        json.dump(all_splits_info, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Dataset {dataset_name} fully processed with all splits")
    return all_splits_info

def main():
    """Main function to execute the sharded HDF5 conversion process with data splitting."""
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
    
    # Sharding and splitting configuration
    SHARD_SIZE = int(config.get("shard_size"))
    TRAIN_RATIO = float(config.get("train_ratio"))
    VAL_RATIO = float(config.get("val_ratio"))
    TEST_RATIO = float(config.get("test_ratio"))
    RANDOM_SEED = int(config.get("random_seed"))
    
    # For large datasets, optimize batch size for I/O performance
    if BATCH_SIZE > 200:
        logging.info(f"Reducing batch size from {BATCH_SIZE} to 200 for better I/O performance")
        BATCH_SIZE = 200

    # Ensure output directory exists
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging_file_name = "hdf5_conversion_errors.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)
    logging.info("Starting HDF5 conversion with train/val/test splits.")
    logging.info(f"Using {N_JOBS} parallel jobs for processing")
    logging.info(f"Batch size: {BATCH_SIZE} files per batch")
    logging.info(f"Splitting data: Train {TRAIN_RATIO:.1%}, Validation {VAL_RATIO:.1%}, Test {TEST_RATIO:.1%}")
    logging.info(f"Using sharded approach with {SHARD_SIZE} files per shard")

    for BASE_DIR in tqdm(list(INPUT_DIRECTORY.iterdir()), desc="Processing datasets", unit="dataset"):
        try:
            if BASE_DIR.is_dir():
                CSV_FILE = BASE_DIR / "metadata.csv"
                if not CSV_FILE.exists():
                    logging.warning(f"Metadata CSV not found for {BASE_DIR.name}, skipping")
                    continue
                
                logging.info(f"Processing dataset: {BASE_DIR.name}")
                # Process with train/val/test splits
                split_info = process_dataset_with_splits(
                    BASE_DIR, CSV_FILE, OUTPUT_DIRECTORY, 
                    SHARD_SIZE, TARGET_SR, MAX_LENGTH_SECONDS, N_JOBS, BATCH_SIZE,
                    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
                )
                
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing dataset {BASE_DIR}: {e}")
            logger.removeHandler(error_file_handler)
            continue

    logging.info("HDF5 conversion with data splitting completed.")

if __name__ == "__main__":
    main()