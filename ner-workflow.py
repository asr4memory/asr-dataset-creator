from gliner import GLiNER
from pathlib import Path 
import logging
import warnings
from tqdm import tqdm
import json
import transformers
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.captureWarnings(True)

# Define input and output directories
input_dir = Path("/home/kompiel/python_scripts/asr-dataset-creator/data/ner_workflow/_input")
output_dir = Path("/home/kompiel/python_scripts/asr-dataset-creator/data/ner_workflow/_output")
output_dir.mkdir(parents=True, exist_ok=True)

# Search for text files (automated transcriptions) in the input directory
transcription_txt_files = list(input_dir.glob("*.txt"))

logging.info(f"Found transcription file: {transcription_txt_files[0]}")

# START NER WORKFLOW

logging.info(f"Start processing {transcription_txt_files[0].name}")

# Load the transcription text from the TXT file
with open(transcription_txt_files[0], "r", encoding="utf-8") as file:
    transcription_text = file.read()

# Split transcription text into lines
lines = transcription_text.splitlines()

# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# Labels for entity prediction
labels = ["Person names", "Residential address"]

# Variable to hold all predicted entities
entities = []

# Variable to hold all warnings
all_warnings = []

# Define batch size (number of lines to process at a time)
batch_size = 15
logging.info(f"Batch size: {batch_size}")

# Calculate the total number of batches
total_batches = (len(lines) + batch_size - 1) // batch_size
logging.info(f"Total batches: {total_batches}")

# Iterate over the lines in chunks of n lines with a progress bar
for i in tqdm(range(0, len(lines), batch_size), total=total_batches, desc="Processed batches"):
    # Combine the lines into a single string
    batch_text = "\n".join(lines[i:i+batch_size])

    # Catch warnings that occur during the prediction
    with warnings.catch_warnings(record=True) as w_list:
        # Force warnings to be displayed as exceptions
        warnings.simplefilter("always")
        
        # Perform entity prediction on the batch
        batch_entities = model.predict_entities(batch_text, labels, threshold=0.5)
        
        # Add the predicted entities to the list
        entities.extend(batch_entities)
            
        # When warnings occur, add them to the list of all warnings
        all_warnings.extend(w_list)

# After processing all batches, log the warnings
if all_warnings:
    logging.warning(f"{len(all_warnings)} warnings occurred during the process:")
    for warning in all_warnings:
        logging.warning(f"{warning.message}")
else:
    logging.info("No warnings occurred during the process.")

# Remove duplicates from the entities list
unique_entities = []
seen_entities = set()  # Set to store unique entities

for entity in entities:
    # Use a tuple of text and label as the identifier for uniqueness
    identifier = (entity["text"], entity["label"])

    if identifier not in seen_entities:
        unique_entities.append(entity)
        seen_entities.add(identifier)

logging.info(f"Total number of found entities before deletion of duplicates: {len(entities)}")
logging.info(f"Total number of found entities after deletion of duplicates: {len(unique_entities)}")  

print(unique_entities)

# Output entities as tab-separated CSV
output_entities_file = output_dir / f"{transcription_txt_files[0].stem}_entities_batch_size={batch_size}.csv"
with open(output_entities_file, "w", encoding="utf-8") as file:
    # Write header
    file.write("entity name\tentity type\n")
    # Write each entity in tab-separated format
    for entity in unique_entities:
        file.write(f"{entity['text']}\t{entity['label']}\n")
logging.info(f"Entities saved as: {output_entities_file}")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

prompt_system = (
    "Du erhältst eine Liste von Entitäten (Personen und Adressen), die per Named Entity Recognition erkannt wurden. "
    "Bitte prüfe, welche der genannten Entitäten historisch sind, und gib die Antwort ausschließlich im JSON-Format zurück. "
    "Jede Entität sollte als ein Objekt im folgenden Format dargestellt werden:\n"
    "{\n"
    '  "entity_name": "<Name der Entität>",\n'
    '  "entity_type": "<Typ der Entität>",\n'
    '  "is_historical": true/false\n'
    "}\n"
    "Gib ausschließlich gültiges JSON zurück. Keine zusätzlichen Kommentare oder Erklärungen."
)

prompt_content = ""
for entity in unique_entities:
    prompt_content += f"- {entity['text']} ({entity['label']})\n"

messages = [
    {"role": "system", "content": prompt_system},
    {"role": "user", "content": prompt_content},
]

outputs = pipeline(messages, max_new_tokens=10000)
output = outputs[0]["generated_text"]

print (output)

assistant_dict = output[2]
assistant_content = assistant_dict["content"]

print (assistant_content)

try:
    parsed_output = json.loads(assistant_content)
except json.JSONDecodeError as e:
    print("Error while loading JSON output:", e)
    parsed_output = {}

output_historical_entities_file = output_dir / f"{transcription_txt_files[0].stem}_historical_entities.json"
with open(output_historical_entities_file, "w", encoding="utf-8") as f:
    json.dump(parsed_output, f, ensure_ascii=False, indent=2)

print(f"Job finished")