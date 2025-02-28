import os
import glob
import librosa

import os
import glob
import h5py
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/dataset-merger/_output/eg_dataset_complete"
DATA_DIR = os.path.join(BASE_DIR, "data")     
CSV_FILE = os.path.join(BASE_DIR, "metadata.csv") 
OUTPUT_HDF5 = "/Users/pkompiel/python_scripts/asr4memory/asr-dataset-creator/data/hdf-converter/_output/eg_dataset_complete_13sec.h5"

TARGET_SR = 16000  
PERCENTILE = 99

# def find_percentile(sr=TARGET_SR, percentile=PERCENTILE):
#     audio_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.wav")))
#     durations = [] 

#     for file in tqdm(audio_files, desc="Calculating durations", unit="file"):
#         audio, sr = librosa.load(file, sr=None)  
#         duration = len(audio) / sr
#         durations.append(duration)

#     durations = np.array(durations)
#     max_lenght_seconds = np.percentile(durations, percentile)
#     max_lenght_samples = int(max_lenght_seconds * sr)
#     print(f"Percentile {percentile} der Audiolängen in Sekunden: {max_lenght_seconds}")
#     print(f"Percentile {percentile} in Samples: {max_lenght_samples}")
    

#     return max_lenght_samples



def hdfconverter(max_length_samples, sr=TARGET_SR):

    df = pd.read_csv(CSV_FILE)

    file_names = df["file_name"].tolist()

    # Create HDF5 file
    with h5py.File(OUTPUT_HDF5, "w") as hdf5_file:
        # dataset for audio waveforms
        # we use "float32" and a fixed size = (number, max_length_samples)
        audio_dataset = hdf5_file.create_dataset(
            "audio_waveforms",
            shape=(len(file_names), max_length_samples),
            dtype="float32",
            compression="gzip",         # or "lzf"
            compression_opts=4,           # change according to need
            chunks=True                 # for better performance on partial access
        )

        # Dataset für die tatsächlichen Längen
        lengths_dataset = hdf5_file.create_dataset(
            "lengths",
            shape=(len(file_names),),
            dtype="int32",
            compression="gzip",         # or "lzf"
            compression_opts=4,           # change according to need
            chunks=True                 # for better performance on partial access
        )

        # Save the metadata in separate datasets
        # Variable length (strings) require special_dtype(vlen=str)
        dt_str = h5py.special_dtype(vlen=str)
        
        transcript_dataset = hdf5_file.create_dataset(
            "transcription",
            shape=(len(file_names),),
            dtype=dt_str,
            compression="gzip",         # or "lzf"
            compression_opts=4,           # change according to need
            chunks=True                 # for better performance on partial access
        )
        
        # Optional: Save the file names
        filename_dataset = hdf5_file.create_dataset(
            "file_name", 
            shape=(len(file_names),),
            dtype=dt_str,
            compression="gzip",          # or "lzf"
            compression_opts=4,           # change according to need
            chunks=True                 # for better performance on partial access
        )

        # Iterate over the rows of the DataFrame
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Converting to HDF5", unit="file"):
            wav_file = row["file_name"]
            transcript = row.get("transcription", "")        # dito

            # Path to the audio file 
            wav_path = os.path.join(BASE_DIR, wav_file)
            if not os.path.isfile(wav_path):
                print(f"Datei existiert nicht: {wav_path}")
                continue

            # Audio laden
            # Anm.: sr=TARGET_SR -> wenn != None, wird resamplt
            audio, sr = librosa.load(wav_path, sr=sr)
            length_samples = len(audio)

            # Falls du Padding möchtest:
            if length_samples > max_length_samples:
                audio = audio[:max_length_samples]  # Abschneiden
                length_samples = max_length_samples
            else:
                padded = np.zeros(max_length_samples, dtype=np.float32)
                padded[:length_samples] = audio
                audio = padded

            # In HDF5 speichern
            audio_dataset[i] = audio
            lengths_dataset[i] = length_samples

            transcript_dataset[i] = str(transcript)
            filename_dataset[i] = wav_file

    print(f"Finished conversion to HDF5: {OUTPUT_HDF5}")


def main():

    # Längstes Audio finden
    # max_length_samples = find_percentile(TARGET_SR, PERCENTILE)

    # Konvertierung in HDF5
    hdfconverter(max_length_samples=208000)


if __name__ == "__main__":
    main()