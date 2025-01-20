from gliner import GLiNER
from pathlib import Path 
import logging
import warnings
from tqdm import tqdm
import json
import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from mlx_lm import load, generate
import torch
from app_config import get_config
import gc
from utils import set_up_logging

"""
To-Do:
- Experiment with different temperatures for LLM filtering
- Compare number of entities in CSV and JSON files --> CHECK
- If there are entites missing in the JSON file, add them from the CSV file with historical flag set to false --> CHECK
- Other Option: Retry LLM filtering until all entities are present in the JSON file (maybe with another temperature value???) --> CHECK
- Remove common wrongly recognized entities from unique_entities list 
- Try running Llama-3.3-70B-Instruct model
"""

# Load the configuration
config = get_config()["ner_workflow"]
config_logging = get_config()["logging"]

INPUT_DIR = Path(config["input_directory"])
OUTPUT_DIR = Path(config["output_directory"])
LOGGING_DIRECTORY = Path(config_logging["logging_directory"])

batch_size = config["ner_batch_size"]
threshold = config["ner_threshold"]
max_new_tokens = config["llm_max_new_tokens"]

def load_gliner_model():
    """
    Load the GLiNER model
    """
    # model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

    return model

def ner_workflow(transcription_txt_file, model):
    """
    Perform Named Entity Recognition (NER) on the transcription
    """
    
    # Load the transcription text from the TXT file
    with open(transcription_txt_file, "r", encoding="utf-8") as file:
        transcription_text = file.read()

    # Split transcription text into lines
    lines = transcription_text.splitlines()

    # Labels for entity prediction
    labels = ["person", "residential address"]

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
    
    # Remove common wrongly recognized entities from unique_entities list
    blacklist = {"sie", "er", "wir", "ich", "du", "ihr", "mein vater", "meine mutter", "meinem vater", "meiner mutter", "meine tochter", "mein sohn", "meiner tochter", "meinem sohn", "mein bruder", "meine schwester", "meinem bruder", "meiner schwester", "mein onkel", "meine tante", "meinem onkel", "meiner tante", "mein opa", "meine oma", "meinem opa", "meiner oma", "mein vetter", "meine cousine", "meinem vetter", "meiner cousine", "mein neffe", "meine nichte", "meinem neffe", "meiner nichte", "mein enkel", "meine enkelin", "meinem enkel", "meiner enkelin", "mein schwager", "meine schwägerin", "meinem schwager", "meiner schwägerin", "mein schwiegersohn", "meine schwiegertochter", "meinem schwiegersohn", "meiner schwiegertochter", "mein schwiegervater", "meine schwiegermutter", "meinem schwiegervater", "meiner schwiegermutter", "mein schwiegersohn", "meine schwiegertochter", "meinem schwiegersohn", "meiner schwiegertochter", "mein schwiegervater", "meine schwiegermutter", "meinem schwiegervater", "meiner schwiegermutter", "mein schwiegersohn", "meine schwiegertochter", "meinem schwiegersohn", "meiner schwiegertochter", "mein schwiegervater", "meine schwiegermutter", "meinem schwiegervater", "meiner schwiegermutter", "mein schwiegersohn", "meine schwiegertochter", "meinem schwiegersohn", "meiner schwiegertochter", "mein schwiegervater", "meine schwiegermutter", "meinem schwiegervater", "meiner schwiegermutter", "mein schwiegersohn", "meine schwiegertochter", "meinem schwiegersohn", "meiner schwiegertochter", "mein schwiegervater", "meine frau", "meiner frau", "mein mann" "meinem mann", "mann", "frau", "mich", "dich", "ihm", "ihr", "ihnen", "papa", "mama", "herr", "sohn", "tochter"}

    # Remove duplicates from the entities list
    unique_entities = []
    seen_entities = set()  # Set to store unique entities
    unique_entities_names= []

    for entity in entities:
        # Skip entities in the blacklist
        if entity["text"].lower() in blacklist:
            continue

        # Use a tuple of text and label as the identifier for uniqueness
        identifier = (entity["text"], entity["label"])

        if identifier not in seen_entities:
            unique_entities.append(entity)
            unique_entities_names.append(entity["text"])
            seen_entities.add(identifier)

    logging.info(f"Total number of found entities before deletion of duplicates: {len(entities)}")
    logging.info(f"Total number of found entities after deletion of duplicates: {len(unique_entities)}")  

    # Output entities as tab-separated CSV

    return unique_entities, unique_entities_names

def save_entities_csv(unique_entities, transcription_txt_file):
    """
    Output entities as tab-separated CSV
    """
    output_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_entities_batch_size={batch_size}.csv"
    with open(output_entities_file, "w", encoding="utf-8") as file:
        # Write header
        file.write("entity name\tentity type\tscore\n")
        # Write each entity in tab-separated format
        for entity in unique_entities:
            file.write(f"{entity['text']}\t{entity['label']}\t{entity['score']}\n")
    logging.info(f"Entities saved as: {output_entities_file}")


def load_llm_model():
    """
    Load the LLM model
    """
    model, tokenizer = load("mlx-community/Llama-3.3-70B-Instruct-4bit")
    return model, tokenizer


def filter_historical_entities(llm_model, llm_tokenizer, unique_entities, unique_entities_names):
    """
    Filter historical entities using a large language model
    """
    # Define the prompt for the LLM
    prompt = (
        "Du erhältst eine Liste von Entitäten (Personen und Adressen), die per Named Entity Recognition erkannt wurden. "
        "Bitte prüfe, bei welchen der genannten Entitäten es sich um historische Persönlichkeiten oder Orte handelt, und gib die Antwort ausschließlich im JSON-Format zurück. "
        "Jede Entität sollte als ein Objekt im folgenden Format dargestellt werden:\n"
        "{\n"
        '  "entity_name": "<Name der Entität>",\n'
        '  "entity_type": "<Typ der Entität>",\n'
        '  "is_historical": true/false\n'
        "}\n"
        "Gib ausschließlich gültiges JSON zurück. Keine zusätzlichen Kommentare oder Erklärungen.\n"
    )
    for entity in unique_entities:
        prompt += f"- {entity['text']} ({entity['label']})\n"
    logging.info("User prompt:")
    print(prompt)


    if hasattr(llm_tokenizer, "apply_chat_template") and llm_tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


    # Generate the output using the LLM
    logging.info("Starting the filtering process with the LLM.")
    response = generate(llm_model, llm_tokenizer, prompt=prompt, verbose=True, max_tokens=10000)
    print(response)

    # Parse the assistant content as JSON
    try:
        parsed_llm_entities = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the JSON output: {e}")
        parsed_llm_entities = []
    
    if not isinstance(parsed_llm_entities, list):
        logging.warning("LLM output was not a list of dictionaries. Retrying the LLM filtering process.")
        parsed_llm_entities = []
        llm_entity_names = []
    
    else:
        llm_entity_names = []
        for entity in parsed_llm_entities:
            llm_entity_names.append(entity["entity_name"])
        
        if set(unique_entities_names) == set(llm_entity_names):
            logging.info("Success: The gliner and LLM outputs contain the same entites")
        else:
            logging.warning("Warning: The gliner and LLM outputs do not contain the same entites. Retrying the LLM filtering process.")

    return parsed_llm_entities, llm_entity_names


def save_historical_entities(parsed_llm_entities, transcription_txt_file):
    # Save the historical entities as JSON
    output_historical_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_historical_entities.json"
    with open(output_historical_entities_file, "w", encoding="utf-8") as f:
        json.dump(parsed_llm_entities, f, ensure_ascii=False, indent=2)
    logging.info(f"Historical entities saved as: {output_historical_entities_file}")


def main():
    """
    Main function to execute the NER and filtering workflow.
    """
    # Set up logging
    logging_file_name = "ner_workflow_errors.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)
    logging.info("Starting NER and filtering workflow.")

    # Search for text files (automated transcriptions) in the input directory
    transcription_txt_files = list(INPUT_DIR.glob("*.txt"))
    logging.info(f"Found transcription files: {transcription_txt_files}")

    # Load the GLiNER model
    gliner_model = load_gliner_model()

    # Load the LLM model
    llm_model, llm_tokenizer = load_llm_model()

    # Start the workflow for NER and filtering historical entities
    for transcription_txt_file in transcription_txt_files:
        try:
            # Start workflow for NER
            logging.info(f"Start processing {transcription_txt_file.name}")
            unique_entities, unique_entities_names = ner_workflow(transcription_txt_file, gliner_model)
        
            # Output entities as tab-separated CSV        
            save_entities_csv(unique_entities, transcription_txt_file)
            
            # Start workflow for filtering historical entities via LLMs
            trials = 0
            llm_entity_names = []
            while set(unique_entities_names) != set(llm_entity_names) and trials < 5:
                parsed_llm_entities, llm_entity_names = filter_historical_entities(llm_model, llm_tokenizer, unique_entities, unique_entities_names)
                trials += 1
            
            if trials == 5:
                logger = logging.getLogger()
                logger.addHandler(error_file_handler)
                logging.error(f"Failed to get the same entities from the LLM for {transcription_txt_file.name} after {trials} trials.")
                logger.removeHandler(error_file_handler)
                continue
            
            # Save the historical entities as JSON
            save_historical_entities(parsed_llm_entities, transcription_txt_file)
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing file {transcription_txt_file.name}: {e}")
            logger.removeHandler(error_file_handler)
            continue
        

    # Clear the GPU memory and delete the pipeline
    torch.cuda.empty_cache()
    gc.collect()

    logging.info("NER and filtering workflow completed.")

if __name__ == "__main__":
    main()