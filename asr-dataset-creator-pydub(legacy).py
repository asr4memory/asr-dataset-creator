import webvtt
from pathlib import Path
import csv
from pydub import AudioSegment

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
    start_time = caption.start_in_seconds
    end_time = caption.end_in_seconds
    text = caption.text
    vtt_segments.append({"start": start_time, "end": end_time, "text": text})

# Load the audio file
audio = AudioSegment.from_file(audio_file)

# Initialize the metadata file
metadata_file = output_folder / audio_filename_stem / "metadata.csv"
with metadata_file.open(mode='w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["file_name", "transcription"])

    # Create and save the audio segments based on the timestamps in the VTT file
    for i, segment in enumerate(vtt_segments):
        start_time = segment['start'] * 1000  # Milliseconds
        end_time = segment['end'] * 1000
        
        audio_chunk = audio[start_time:end_time]
        
        # Downsampling the audio to 16 kHz
        audio_chunk = audio_chunk.set_frame_rate(16000)
        
        # Create the file name for the audio segment
        segment_filename = f"{audio_filename_stem}_audio_segment_{i+1}.wav"
        segment_path = audio_filename_stem / data_folder / segment_filename
        
        # Save the audio segment
        audio_chunk.export(segment_path, format="wav")
        
        # Add the segment to the metadata file
        csvwriter.writerow([segment_path.relative_to(output_folder / audio_filename_stem), segment['text']])

print("Audio segments and metadata file have been successfully created.")