import os
import glob
import librosa
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/dataset-merger/_output/eg_dataset_complete"
DATA_DIR = os.path.join(BASE_DIR, "data")        
CSV_FILE = os.path.join(BASE_DIR, "metadata.csv") 
OUTPUT_HDF5 = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/hdf-converter/_output/eg_dataset_complete_vlen.h5"

TARGET_SR = 16000  

def hdfconverter_vlen(sr=TARGET_SR):
    df = pd.read_csv(CSV_FILE)
    file_names = df["file_name"].tolist()
    
    # 2.  Create HDF5 file
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_file:
        # Create a variable-length datatype for audio waveforms
        dt_vlen = h5py.vlen_dtype(np.dtype("float32"))
        
        # Create a dataset for audio waveforms
        audio_dataset = hdf5_file.create_dataset(
            "audio_waveforms",
            shape=(len(file_names),),
            dtype=dt_vlen
        )
        
        # Create a dataset for the actual lengths
        lengths_dataset = hdf5_file.create_dataset(
            "lengths",
            shape=(len(file_names),),
            dtype="int32"
        )
        
        # For metadata, we use a variable-length string datatype
        dt_str = h5py.special_dtype(vlen=str)
        transcript_dataset = hdf5_file.create_dataset(
            "transcription",
            shape=(len(file_names),),
            dtype=dt_str
        )
        filename_dataset = hdf5_file.create_dataset(
            "file_name", 
            shape=(len(file_names),),
            dtype=dt_str
        )
        
        # Iterate over the rows of the DataFrame
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Converting to HDF5 (vlen)", unit="file"):
            wav_file = row["file_name"]
            transcript = row.get("transcription", "")
            
            # Path to the audio file
            wav_path = os.path.join(BASE_DIR, wav_file)
            if not os.path.isfile(wav_path):
                print(f"Datei existiert nicht: {wav_path}")
                continue
            
            # Load the audio file
            audio, _ = librosa.load(wav_path, sr=sr)
            length_samples = len(audio)
            
            # Save the audio waveform, length, transcript, and filename
            audio_dataset[i] = audio
            lengths_dataset[i] = length_samples
            
            transcript_dataset[i] = str(transcript)
            filename_dataset[i] = wav_file

    print(f"Finished conversion to HDF5 (vlen): {OUTPUT_HDF5}")

def main():
    hdfconverter_vlen(sr=TARGET_SR)

if __name__ == "__main__":
    main()
