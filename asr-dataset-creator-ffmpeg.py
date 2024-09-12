import webvtt
from pathlib import Path
import csv
import subprocess

# Definieren Sie die Eingabe- und Ausgabeordner
input_folder = Path('/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_input')
output_folder = Path('/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output')

# Stellen Sie sicher, dass der Ausgabeordner existiert
output_folder.mkdir(parents=True, exist_ok=True)

# Suchen Sie nach einer WAV- und einer VTT-Datei im Eingabeordner
wav_files = list(input_folder.glob('*.wav'))
vtt_files = list(input_folder.glob('*.vtt'))

# Stellen Sie sicher, dass jeweils genau eine WAV- und eine VTT-Datei gefunden wird
if len(wav_files) != 1 or len(vtt_files) != 1:
    raise ValueError("Es muss genau eine WAV- und eine VTT-Datei im Eingabeordner vorhanden sein.")

audio_file = wav_files[0]
vtt_file = vtt_files[0]

# Dateinamen ohne Pfad und Suffix extrahieren
audio_filename_stem = audio_file.stem

# Erstellen Sie einen Unterordner für die Audiodaten und stellen Sie sicher, dass er existiert
data_folder = output_folder / audio_filename_stem / 'data'
data_folder.mkdir(parents=True, exist_ok=True)

# VTT-Datei parsen
vtt_segments = []

for caption in webvtt.read(vtt_file):
    start_time = caption.start_in_seconds
    end_time = caption.end_in_seconds
    text = caption.text
    vtt_segments.append({"start": start_time, "end": end_time, "text": text})

# Metadata CSV-Datei initialisieren
metadata_file = output_folder / audio_filename_stem / "metadata.csv"
with metadata_file.open(mode='w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["file_name", "transcription"])

    # Erstellen und speichern der Segmente basierend auf den VTT-Zeitstempeln
    for i, segment in enumerate(vtt_segments):
        start_time = segment['start']
        end_time = segment['end']
        
        # Datei-Namen für das Segment erstellen
        segment_filename = f"{audio_filename_stem}_audio_segment_{i+1}.wav"
        segment_path = audio_filename_stem / data_folder / segment_filename
            
        # Berechnen der Segmentdauer
        duration = end_time - start_time
            
        # FFmpeg-Befehl zum Schneiden und Heruntersampeln des Audios
        ffmpeg_command = [
            'ffmpeg', '-i', str(audio_file),
            '-ss', f"{start_time:.3f}", '-t', f"{duration:.3f}",
            '-ar', '16000',
            str(segment_path)
        ]
            
        # FFmpeg-Befehl ausführen
        subprocess.run(ffmpeg_command, check=True)
            
        # Zeile zur CSV-Datei hinzufügen
        csvwriter.writerow([segment_path.relative_to(output_folder / audio_filename_stem), segment['text']])

print("Audio-Segmente und Metadata-Datei erfolgreich erstellt und gespeichert.")