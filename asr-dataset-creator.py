import webvtt
from pydub import AudioSegment

# VTT-Datei laden und parsen
vtt_file = 'transcript.vtt'
vtt_segments = []

for caption in webvtt.read(vtt_file):
    start_time = caption.start_in_seconds
    end_time = caption.end_in_seconds
    text = caption.text
    vtt_segments.append({"start": start_time, "end": end_time, "text": text})

# Audio laden
audio = AudioSegment.from_file("audiofile.wav")

# Erstellen und speichern der Segmente basierend auf den VTT-Zeitstempeln
for i, segment in enumerate(vtt_segments):
    start_time = segment['start'] * 1000  # Millisekunden
    end_time = segment['end'] * 1000
    
    audio_chunk = audio[start_time:end_time]
    
    # Speichern Sie die einzelnen Audio-Segmente
    audio_chunk.export(f"audio_segment_{i+1}.wav", format="wav")

print("Audio-Segmente erfolgreich erstellt und gespeichert.")