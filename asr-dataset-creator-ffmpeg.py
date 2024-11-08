import webvtt
from pathlib import Path
import csv
import subprocess
from tqdm import tqdm

# Define the input and output folders
input_folder = Path('/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_input')
output_folder = Path('/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output')

# Check if the output folder exists, if not create it
output_folder.mkdir(parents=True, exist_ok=True)

# Search for WAV and VTT files in the input folder
wav_files = list(input_folder.glob('*.wav'))
vtt_files = list(input_folder.glob('*.vtt'))

# Check if there is exactly one WAV and one VTT file in the input folder
if len(wav_files) != 1 or len(vtt_files) != 1:
    raise ValueError("There has to be exactly one WAV and one VTT file in the input folder.")

audio_file = wav_files[0]
vtt_file = vtt_files[0]

# Extract the audio file name stem
audio_filename_stem = audio_file.stem

# Create a folder for the audio file and check if it already exists, if not create it
data_folder = output_folder / audio_filename_stem / 'data'
data_folder.mkdir(parents=True, exist_ok=True)

# Parse the VTT file
vtt_segments = []

for caption in webvtt.read(vtt_file):
    start_time = caption.start  # Use original VTT format with milliseconds
    end_time = caption.end      # Use original VTT format with milliseconds
    text = caption.text
    vtt_segments.append({"start": start_time, "end": end_time, "text": text})

# Initialize the metadata file
metadata_file = output_folder / audio_filename_stem / "metadata.csv"

# Check if metadata file exists and load existing entries
existing_files = set()
if metadata_file.exists():
    with metadata_file.open(mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        existing_files = {row[0] for row in csvreader}  # Collect existing file names

# Open metadata file in append mode
with metadata_file.open(mode='a', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header only if the file is empty
    if not existing_files:
        csvwriter.writerow(["file_name", "transcription"])

    print("Start processing audio segments...")

    # Create and save the audio segments based on the timestamps in the VTT file
    for i, segment in enumerate(tqdm(vtt_segments, desc="Processing audio segments", unit="segment")):
        start_time = segment['start']
        end_time = segment['end']
        
        # Create the file name for the audio segment
        segment_filename = f"{audio_filename_stem}_audio_segment_{i+1}.wav"
        segment_path = audio_filename_stem / data_folder / segment_filename

        # Check if the segment file already exists; if so, skip it
        if segment_path.exists():
            continue
            
        # FFmpeg command to extract the audio segment using precise start and end times
        ffmpeg_command = [
            'ffmpeg', '-loglevel', 'error', '-i', str(audio_file),
            '-ss', start_time, '-to', end_time,
            '-ar', '16000',
            str(segment_path)
            ]
            
        # Run the FFmpeg command
        subprocess.run(ffmpeg_command, check=True)
            
        # Add the segment as a new row to the metadata file
        csvwriter.writerow([segment_path.relative_to(output_folder / audio_filename_stem), segment['text']])

print("Audio segments and metadata file have been successfully created.")