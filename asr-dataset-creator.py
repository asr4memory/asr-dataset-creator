import webvtt
from pathlib import Path
import csv
import subprocess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from app_config import get_config

# Load the configuration
config = get_config()["dataset_creator"]

INPUT_FOLDER = Path(config["input_directory"])
OUTPUT_FOLDER = Path(config["output_directory"])
SAMPLE_RATE = config["sample_rate"]
OFFSET = config["offset"]


def parse_vtt_file(vtt_file):
    """Parse the VTT file and extract segments with start, end times, and text."""
    segments = [
        {"start": caption.start, "end": caption.end, "text": caption.text}
        for caption in webvtt.read(vtt_file)
    ]
    return segments


def setup_output_structure(output_folder, audio_filename_stem):
    """Prepare the output directories and metadata file path."""
    data_folder = output_folder / audio_filename_stem / 'data'
    data_folder.mkdir(parents=True, exist_ok=True)
    metadata_file = output_folder / audio_filename_stem / "metadata.csv"
    return data_folder, metadata_file


def load_existing_metadata(metadata_file):
    """Load existing metadata entries to avoid redundant processing."""
    if metadata_file.exists():
        with metadata_file.open(mode='r', encoding='utf-8') as file:
            csvreader = csv.reader(file)
            next(csvreader, None)  # Skip header
            return {row[0] for row in csvreader}  # Collect existing file names
    return set()


def time_to_seconds(time_str):
    """Convert a time string in HH:MM:SS.sss format to seconds."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def process_segment(args):
    """Extract a segment from the audio file using FFmpeg."""
    (i, segment), audio_file, data_folder, output_folder, audio_filename_stem = args
    segment_filename = f"{audio_filename_stem}_audio_segment_{i+1}.wav"
    segment_path = data_folder / segment_filename

    # Skip if the segment file already exists
    if segment_path.exists():
        return None
    
    adjusted_start = max(0, time_to_seconds(segment['start']) + OFFSET)
    adjusted_end = time_to_seconds(segment['end']) + OFFSET

    # FFmpeg command to extract the segment
    ffmpeg_command = [
        'ffmpeg', '-loglevel', 'error', '-i', str(audio_file),
        '-ss', str(adjusted_start), '-to', str(adjusted_end),
        '-ar', SAMPLE_RATE, str(segment_path)
    ]
    subprocess.run(ffmpeg_command, check=True)
    return segment_path.relative_to(output_folder / audio_filename_stem), segment['text']


def main():
    # Verify and locate input files
    wav_files = list(INPUT_FOLDER.glob('*.wav'))
    vtt_files = list(INPUT_FOLDER.glob('*.vtt'))
    if len(wav_files) != 1 or len(vtt_files) != 1:
        raise ValueError("There must be exactly one WAV and one VTT file in the input folder.")

    audio_file = wav_files[0]
    vtt_file = vtt_files[0]
    audio_filename_stem = audio_file.stem

    # Prepare output structure and load existing metadata entries
    data_folder, metadata_file = setup_output_structure(OUTPUT_FOLDER, audio_filename_stem)
    existing_files = load_existing_metadata(metadata_file)
    vtt_segments = parse_vtt_file(vtt_file)

    # Prepare tasks for parallel processing
    tasks = [
        ((i, segment), audio_file, data_folder, OUTPUT_FOLDER, audio_filename_stem)
        for i, segment in enumerate(vtt_segments)
        if f"{audio_filename_stem}_audio_segment_{i+1}.wav" not in existing_files
    ]

    # Process segments in parallel
    print("Start processing audio segments...")
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_segment, tasks), total=len(tasks),
                            desc="Processing audio segments", unit="segment"))

    # Write new metadata entries
    with metadata_file.open(mode='a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        if not existing_files:
            csvwriter.writerow(["file_name", "transcription"])
        csvwriter.writerows(result for result in results if result)

    print("Audio segments and metadata file have been successfully created.")


if __name__ == "__main__":
    main()