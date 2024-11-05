from datasets import load_dataset, DatasetDict

# Laden des Datasets von der bereinigten CSV-Datei
dataset = load_dataset("audiofolder", data_dir="/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output/eg085_01_01")

# Dataset in Trainings- und Testsets aufteilen mit festem Seed
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)  # 42 ist ein Beispielwert für den Seed

# Erstellen eines DatasetDict-Objekts
dataset = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test']
})

# Überprüfen des ersten Datensatzes
print(dataset['train'][0])
print(dataset['test'][0])

# Gesamtes DatasetDict anzeigen
print(dataset)