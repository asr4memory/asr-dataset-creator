from datasets import load_dataset, DatasetDict
from app_config import get_config

# Initialize the configuration
config = get_config()["dataset_test"]

# Load the dataset
dataset = load_dataset("audiofolder", data_dir=config["input_directory"])

print("Dataset Structure:", dataset)
print("Keys:", dataset.keys())  # Check for any unexpected split names

# If it’s a DatasetDict, print details of each split
if isinstance(dataset, DatasetDict):
    for split in dataset.keys():
        print(f"Split: {split}, Length: {len(dataset[split])}")
else:
    print(f"Dataset Length: {len(dataset)}")

# Split the dataset into training and testing sets
split_dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)  # 42 ist ein Beispielwert für den Seed

# Create a DatasetDict object
dataset = DatasetDict({
    'train': split_dataset['train'],
    'test': split_dataset['test']
})

# Check the first example in the training and testing sets
print(dataset['train'][0])
print(dataset['test'][0])

# Show the whole dataset
print(dataset)