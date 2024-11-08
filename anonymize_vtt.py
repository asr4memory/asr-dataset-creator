import pandas as pd
import re
from pathlib import Path

# Define input directories and files
input_dir = Path("/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/vtt_anon/_input")
output_dir = Path("/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/vtt_anon/_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Search for vtt files and anonymized word segment CSV files in the input directory
vtt_files = list(input_dir.glob("*.vtt"))
csv_files = list(input_dir.glob("*.csv"))

#Check if there is exactly one VTT and one CSV file in the input directory
if len(vtt_files) != 1 or len(csv_files) != 1:
    raise ValueError("Es muss genau eine TXT- und eine CSV-Datei im Eingabeordner vorhanden sein.")

input_vtt_file = vtt_files[0]
input_csv_file = csv_files[0]

# Step 1: Load the anonymized CSV file
df = pd.read_csv(input_csv_file, delimiter="\t", encoding="utf-8")

# Step 2: Remove punctuation from the 'WORD' column
df['CLEANED_WORD'] = df['WORD'].apply(lambda x: re.sub(r'[^\w\s-]', '', x) if isinstance(x, str) else x)

# Step 3: Find all words with 'x' in the SCORE column
words_to_anonymize = df[df['SCORE'] == 'x']['CLEANED_WORD'].tolist()

# Step 4: Remove duplicates by converting to a set, then back to a list
words_to_anonymize = list(set(words_to_anonymize))

print(f"Words to anonymize: {words_to_anonymize}")

# Step 5: Load the VTT file
with open(input_vtt_file, "r", encoding="utf-8") as vtt_file:
    vtt_content = vtt_file.read()

# Step 6: Replace unnecessary characters in the VTT content
vtt_content = re.sub(r'_', '', vtt_content)
vtt_content = re.sub(r'„', '', vtt_content)
vtt_content = re.sub(r'“', '', vtt_content)
vtt_content = re.sub(r'"', '', vtt_content)
vtt_content = re.sub(r'(?:.*\n){3}.*\(\.\.\.\?\).*', '', vtt_content) # Remove whole segments with (...?)
vtt_content = re.sub(r'\(', '', vtt_content)
vtt_content = re.sub(r'\?\)', '', vtt_content)
vtt_content = re.sub(r'\n{3,}', '\n\n', vtt_content)

# Step 7: Replace each word in the VTT content
for word in words_to_anonymize:
    # Create a regular expression to match the word as a whole word (to avoid partial replacements)
    word_regex = r'\b' + re.escape(word) + r'\b'
    vtt_content = re.sub(word_regex, '', vtt_content)

# Step 8: Save the modified VTT file
output_vtt_file = output_dir / f"{input_vtt_file.stem}_anonymized.vtt"
with open(output_vtt_file, "w", encoding="utf-8") as vtt_file:
    vtt_file.write(vtt_content)

print(f"Anonymized VTT file saved at: {output_vtt_file}")