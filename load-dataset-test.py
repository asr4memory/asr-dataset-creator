from datasets import load_dataset, DatasetDict

# Laden des Datasets von der bereinigten CSV-Datei
dataset = load_dataset("audiofolder", data_dir="/Users/peterkompiel/python_scripts/asr4memory/processing_files/whisper-train/_output")

# Dataset in Trainings- und Testsets aufteilen
split_dataset = dataset['train'].train_test_split(test_size=0.2)  # 20% für Testdaten

# Erstellen eines DatasetDict-Objekts
dataset_dict = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test']
})

# Überprüfen des ersten Datensatzes
print(dataset_dict['train'][0])
print(dataset_dict['test'][0])

# Gesamtes DatasetDict anzeigen
print(dataset_dict)