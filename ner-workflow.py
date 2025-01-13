from gliner import GLiNER
from pathlib import Path 
import logging
import warnings
from tqdm import tqdm
import json
import transformers
import torch
from app_config import get_config
import gc

"""
To-Do:
- Compare number of entities in CSV and JSON files
- Remove common wrongly recognized entities from unique_entities list 
"""

# Load the configuration
config = get_config()["ner_workflow"]

INPUT_DIR = Path(config["input_directory"])
OUTPUT_DIR = Path(config["output_directory"])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

batch_size = config["ner_batch_size"]
threshold = config["ner_threshold"]
max_new_tokens = config["llm_max_new_tokens"]

def ner_workflow(transcription_txt_file):
    """
    Perform Named Entity Recognition (NER) on the transcription
    """
    
    # Load the transcription text from the TXT file
    with open(transcription_txt_file, "r", encoding="utf-8") as file:
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
            batch_entities = model.predict_entities(batch_text, labels, threshold=threshold)
        
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

    # Output entities as tab-separated CSV
    output_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_entities_batch_size={batch_size}.csv"
    with open(output_entities_file, "w", encoding="utf-8") as file:
        # Write header
        file.write("entity name\tentity type\n")
        # Write each entity in tab-separated format
        for entity in unique_entities:
            file.write(f"{entity['text']}\t{entity['label']}\n")
    logging.info(f"Entities saved as: {output_entities_file}")

    return unique_entities


def load_model():
    """
    Load the LLM model
    """
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",   
    )
    return pipeline


def filter_historical_entities(llm_pipeline, unique_entities, transcription_txt_file):
    """
    Filter historical entities using a large language model
    """
    # Define the prompt for the LLM
    prompt_system = (
        "Du erhältst eine Liste von Entitäten (Personen und Adressen), die per Named Entity Recognition erkannt wurden. "
        "Bitte prüfe, bei welchen der genannten Entitäten es sich um historische Persönlichkeiten oder Orte handelt, und gib die Antwort ausschließlich im JSON-Format zurück. "
        "Jede Entität sollte als ein Objekt im folgenden Format dargestellt werden:\n"
        "{\n"
        '  "entity_name": "<Name der Entität>",\n'
        '  "entity_type": "<Typ der Entität>",\n'
        '  "is_historical": true/false\n'
        "}\n"
        "Gib ausschließlich gültiges JSON zurück. Keine zusätzlichen Kommentare oder Erklärungen."
    )
    logging.info("System prompt:")
    print(prompt_system)

    prompt_content = ""
    for entity in unique_entities:
        prompt_content += f"- {entity['text']} ({entity['label']})\n"
    logging.info("User prompt:")
    print(prompt_content)

    # Combine the system and user prompts
    messages = [
        {"role": "system", "content": prompt_system},
        {"role": "user", "content": prompt_content},
    ]
    # Generate the output using the LLM
    logging.info("Starting the filtering process with the LLM.")
    outputs = llm_pipeline(messages, max_new_tokens=5000)
    output = outputs[0]["generated_text"]

    # Extract the assistant content from the output
    assistant_dict = output[2]
    assistant_content = assistant_dict["content"]

    # Parse the assistant content as JSON
    try:
        parsed_output = json.loads(assistant_content)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the JSON output: {e}")
        parsed_output = {}

    # Save the historical entities as JSON
    output_historical_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_historical_entities.json"
    with open(output_historical_entities_file, "w", encoding="utf-8") as f:
        json.dump(parsed_output, f, ensure_ascii=False, indent=2)
    logging.info(f"Historical entities saved as: {output_historical_entities_file}")


def main():
    """
    Main function to execute the NER and filtering workflow.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.captureWarnings(True)

    # Search for text files (automated transcriptions) in the input directory
    transcription_txt_files = list(INPUT_DIR.glob("*.txt"))
    logging.info(f"Found transcription files: {transcription_txt_files}")

    # Load the LLM model
    llm_pipeline = load_model()

    # Start the workflow for NER and filtering historical entities
    for transcription_txt_file in transcription_txt_files:
        # Start workflow for NER
        logging.info(f"Start processing {transcription_txt_file.name}")
        unique_entities = ner_workflow(transcription_txt_file)
        
        # Start workflow for filtering historical entities via LLMs
        filter_historical_entities(llm_pipeline, unique_entities, transcription_txt_file)

    # Clear the GPU memory and delete the pipeline
    torch.cuda.empty_cache()
    gc.collect()

    logging.info("NER and filtering workflow completed.")

if __name__ == "__main__":
    main()