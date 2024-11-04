from datasets import load_dataset, DatasetDict

dataset = DatasetDict()

# Laden des Datasets von der bereinigten CSV-Datei
dataset["train"] = load_dataset("audiofolder", data_dir="/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output/eg085_01_01/", split="train")
dataset["test"] = load_dataset("audiofolder", data_dir="/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output/eg085_01_01/", split="test")

# Überprüfen des ersten Datensatzes
print(dataset['train'][0])
print(dataset['test'][0])

# Gesamtes DatasetDict anzeigen
print(dataset)