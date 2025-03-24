from gliner import GLiNER
from pathlib import Path 
import logging
import warnings
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from thefuzz import fuzz, process
from mlx_lm import load, generate
import torch
from app_config import get_config
import gc
from utils import set_up_logging

"""
To-Do:
- Experiment with different temperatures for LLM filtering
- Compare number of entities in CSV and JSON files --> CHECK
- Retry LLM filtering until all entities are present in the JSON file (maybe with another temperature value???) --> CHECK
- Remove common wrongly recognized entities from unique_entities list  --> CHECK
- Try running Llama-3.3-70B-Instruct model --> CHECK
- Automatically add false names to blacklist? --> CHECK
- Implement Entity Linking with CSV of historical persons --> NEW
"""

def load_gliner_model():
    """
    Load the GLiNER model
    """
    # model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

    return model

def load_historical_entities(historical_entities_file):
    """
    Load the CSV file containing historical entities (both persons and locations)
    """
    try:
        # Try to load the CSV file with historical entities
        historical_df = pd.read_csv(historical_entities_file, sep='\t', engine='python', encoding='UTF-8')
        logging.info(f"Loaded {len(historical_df)} historical entities from {historical_entities_file}")
        
        # Check if the required columns exist
        required_columns = ['name', 'parent']  # Adjust based on your CSV structure
        for col in required_columns:
            if col not in historical_df.columns:
                logging.warning(f"Required column '{col}' not found in historical entities CSV.")
        
        # Count types of historical entities
        if 'parent' in historical_df.columns:
            type_counts = historical_df['parent'].value_counts()
            for entity_type, count in type_counts.items():
                logging.info(f"Found {count} historical entities of type '{entity_type}'")
        
        return historical_df
    except Exception as e:
        logging.error(f"Error loading historical entities file: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    

def ner_workflow(transcription_txt_file, model, threshold, batch_size):
    """
    Perform Named Entity Recognition (NER) on the transcription
    """
    
    # Load the transcription text from the TXT file
    with open(transcription_txt_file, "r", encoding="utf-8") as file:
        transcription_text = file.read()

    # Split transcription text into lines
    lines = transcription_text.splitlines()

    # Labels for entity prediction
    labels = ["person", "full address"] # residential address also possible

    # Variable to hold all predicted entities
    entities = []

    # Variable to hold all warnings
    all_warnings = []

    # Log the number of lines in the transcription
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
    with open('./data/blacklist.txt', 'r', encoding='utf-8') as file:
        blacklist = {line.strip().lower() for line in file if line.strip()}

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

    return unique_entities, unique_entities_names

def save_entities_csv(unique_entities, transcription_txt_file, batch_size, OUTPUT_DIR):
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


def load_llm_model():
    """
    Load the LLM model
    """
    model, tokenizer = load("mlx-community/Llama-3.3-70B-Instruct-4bit")
    return model, tokenizer


def filter_real_entities(llm_model, llm_tokenizer, unique_entities, unique_entities_names, trials, max_new_tokens):
    """
    Filter real entities (names and addresses) using a large language model
    """
    sym_diff = set()
    # Define the prompt for the LLM
    prompt = (
        "Du erhältst eine Liste von Entitäten (Personen und Adressen), die per Named Entity Recognition erkannt wurden.\n"
        "1) Bitte prüfe, bei welchen der genannten Entitäten es sich um historische Persönlichkeiten oder Orte handelt, und gib die Antwort ausschließlich im JSON-Format zurück.\n"
        "2) Bitte prüfe, bei welchen Personen (Entitätentyp 'person') es sich um ausschließlich richtige Vornamen und/oder Nachnamen handelt. Dabei kann es sich entweder nur um einen Vornamen (Beispiel: Max) oder nur um einen Nachnamen (Beispiel: Müller) handeln oder der Vor- und Nachname zusammengeschrieben sein (Beispiel: Max Müller). Die Vor- und Nachnamen können auch in Zusammenhang mit anderen Begriffen stehen (Beispiele: Herr Müller, Frau Müller). Bei dem Entitätentyp 'full address' sollte in diesem Fall der Wert immer 'false' sein.\n"
        "3) Bitte prüfe, bei welchen Adressen (Entitätentyp 'full address') es sich um richtige Adressen handelt. Bei einer richtigen Adresse handelt es sich ausschließlich nur dann, wenn eine Straße zusammen mit einer Hausnummer auftritt (Beispiel: Musterstraße 1). Bei dem Entitätentyp 'person' sollte in diesem Fall der Wert immer 'false' sein.\n"
        "Jede Entität sollte als ein Objekt im folgenden Format dargestellt werden:\n"
        "{\n"
        '  "entity_name": "<Name der Entität>",\n'
        '  "entity_type": "<Typ der Entität>",\n'
        '  "is_historical": true/false\n'
        '  "is_real_name": true/false\n'
        '  "is_real_address": true/false\n'
        "}\n"
        "Gib ausschließlich gültiges JSON zurück. Keine zusätzlichen Kommentare, Erklärungen oder Formatierungszeichen wie Backticks ('''/```).\n"
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
    logging.info(f"Starting the filtering process with the LLM. Trial: {trials}")
    response = generate(llm_model, llm_tokenizer, prompt=prompt, verbose=False, max_tokens=max_new_tokens)
    # print(response)

    # Parse the assistant content as JSON
    try:
        parsed_llm_entities = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing the JSON output: {e}")
        parsed_llm_entities = []
    
    if not isinstance(parsed_llm_entities, list):
        logging.warning("LLM output was not a list of dictionaries.")
        parsed_llm_entities = []
        llm_entity_names = []
    
    else:
        llm_entity_names = []
        for entity in parsed_llm_entities:
            llm_entity_names.append(entity["entity_name"])
        
        if set(unique_entities_names) == set(llm_entity_names):
            logging.info("Success: The gliner and LLM outputs contain the same entites")
            # Add entities with is_real_name: false to blacklist
            with open('./data/blacklist_new_entries.txt', 'a', encoding='utf-8') as file:
                for entity in parsed_llm_entities:
                    if ((entity["is_real_name"] == False and entity["entity_type"] == "person") or
                        (entity["is_real_address"] == False and (entity["entity_type"] == "residential address" or entity["entity_type"] == "full address"))):
                        file.write(f"{entity['entity_name'].lower()}\n") 
        else:
            logging.warning("Warning: The gliner and LLM outputs do not contain the same entites.")
            sym_diff = set(llm_entity_names).symmetric_difference(set(unique_entities_names))
            logging.warning(f"Different elements: {sym_diff}")

    return parsed_llm_entities, llm_entity_names, sym_diff


def save_json_entities(entities, output_entities_file):
    """
    Save the LLM entities as JSON
    """
    with open(output_entities_file, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

def initialize_historical_entities(parsed_llm_entities, unique_entities):
    """
    Initialize the list of historical entities with the real names and addresses
    """
    expanded_entities = []
    for entity in parsed_llm_entities:
        # Find the original entity data with score
        original_entity = next((e for e in unique_entities if e["text"] == entity["entity_name"]), None)
        if original_entity:
            # Create a new dict with all necessary information
            real_entity = {
                "entity_name": entity["entity_name"],
                "entity_type": entity["entity_type"],
                "gliner_score": original_entity.get("score", 0),
                "is_real_name": entity.get("is_real_name", False),
                "is_real_address": entity.get("is_real_address", False),
                "is_historical": entity.get("is_historical", False),
            }
            expanded_entities.append(real_entity)

    return expanded_entities


def entity_linking(entities, historical_df, threshold=80):
    """
    Perform entity linking to historical
    """
    if historical_df.empty:
        logging.warning("Historical entities DataFrame is empty. Skipping entity linking.")
        return entities
    
    # Create dictionaries for historical entities by type
    historical_entities_by_type = {}
    
    if 'parent' in historical_df.columns:
        # Group historical entities by their type
        for entity_type in historical_df['parent'].unique():
            type_filter = historical_df['parent'] == entity_type
            historical_entities_by_type[entity_type] = historical_df[type_filter]['name'].tolist()
    else:
        # If no type column, just use all names
        historical_entities_by_type['default'] = historical_df['name'].tolist()
    
    # Enhance entities with historical information
    enhanced_entities = []
    
    for entity in entities:
        entity_text = entity['entity_name']
        entity_type = entity['entity_type']
        is_real_name = entity['is_real_name']
        is_real_address = entity['is_real_address']
        
        # Initialize historical information if not already present
        if 'is_historical' not in entity:
            entity['is_historical'] = False
        if 'match_found_in_list' not in entity:
            entity['match_found_in_list'] = None
        if 'match_score' not in entity:
            entity['match_score'] = 0
        
        # Map GLiNER entity types to historical entity types
        matching_types = []
        if entity_type == "person" and is_real_name:
            # Look for historical persons
            matching_types.extend([t for t in historical_entities_by_type.keys() 
                                  if 'person' in t.lower() or 'people' in t.lower()])
        elif entity_type == "full address" or entity_type == "residential address" and is_real_address:
            # Look for historical locations
            matching_types.extend([t for t in historical_entities_by_type.keys() 
                                  if 'location' in t.lower() or 'place' in t.lower() or 
                                  'address' in t.lower() or 'ort' in t.lower()])
        
        # If no specific types match, try with all types
        if not matching_types and 'default' in historical_entities_by_type:
            matching_types = ['default']
        elif not matching_types:
            matching_types = list(historical_entities_by_type.keys())
        
        # Create a combined list of all potential matches
        potential_matches = []
        for match_type in matching_types:
            if match_type in historical_entities_by_type:
                potential_matches.extend(historical_entities_by_type[match_type])
        logging.debug(f"Potential matches for '{entity_text}' ({entity_type}): {potential_matches}")
        
        # Only proceed if we have potential matches
        if potential_matches:
            # Find the best match from historical entities
            best_match, score = process.extractOne(entity_text, potential_matches, scorer=fuzz.token_sort_ratio)
            
            if score >= threshold:
                entity['is_historical'] = True
                entity['match_found_in_list'] = best_match
                entity['match_score'] = score
                
                logging.debug(f"Entity '{entity_text}' ({entity_type}) matched to historical entity '{best_match}' with score {score}")
        
        enhanced_entities.append(entity)
    
    logging.info(f"Entity linking completed. {sum(1 for e in enhanced_entities if e['is_historical'])} of {len(enhanced_entities)} entities identified as historical.")
    return enhanced_entities


def main():
    """
    Main function to execute the NER and filtering workflow.
    """
    # Load the configuration
    config = get_config()["ner_workflow"]
    config_logging = get_config()["logging"]

    # Set up the paths
    INPUT_DIR = Path(config["input_directory"])
    OUTPUT_DIR = Path(config["output_directory"])
    LOGGING_DIRECTORY = Path(config_logging["logging_directory"])
    HISTORICAL_ENTITIES_FILE = Path(config["entity_linking_file"])

    # Configuration parameters for NER
    batch_size = config["ner_batch_size"]
    threshold = config["ner_threshold"]

    # Configuration parameter for LLM filtering
    max_new_tokens = config["llm_max_new_tokens"]

    # Configuration parameter for entity linking
    entity_linking_threshold = config["entity_linking_threshold"]
    
    # Make sure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logging_file_name = "ner_workflow_errors.log"
    error_file_handler = set_up_logging(LOGGING_DIRECTORY, logging_file_name)
    logging.info("Starting NER and filtering workflow.")

    # Search for text files (automated transcriptions) in the input directory
    transcription_txt_files = list(INPUT_DIR.glob("*.txt"))
    logging.info(f"Found transcription files: {transcription_txt_files}")

    # Load the GLiNER model
    gliner_model = load_gliner_model()
    
    # Load the historical entities CSV
    historical_df = load_historical_entities(HISTORICAL_ENTITIES_FILE)

    # Load the LLM model
    llm_model, llm_tokenizer = load_llm_model()

    # Start the workflow for NER and filtering historical entities
    for transcription_txt_file in tqdm(transcription_txt_files, desc="Processed transcription files", unit="file"):
        try:
            # Start workflow for NER
            logging.info(f"Start processing {transcription_txt_file.name}")
            unique_entities, unique_entities_names = ner_workflow(transcription_txt_file, gliner_model, threshold, batch_size)
        
            # Output GliNER entities as tab-separated CSV
            save_entities_csv(unique_entities, transcription_txt_file, batch_size, OUTPUT_DIR)
            logging.info(f"GliNER entities saved as CSV for: {transcription_txt_file.name}")
            
            # First use LLM to filter real names and addresses
            trials = 1
            llm_entity_names = []
            parsed_llm_entities = []
            
            while set(unique_entities_names) != set(llm_entity_names) and trials <= 1:
                parsed_llm_entities, llm_entity_names, sym_diff = filter_real_entities(llm_model, llm_tokenizer, unique_entities, unique_entities_names, trials, max_new_tokens)
                trials += 1
                if sym_diff:
                    with open('./data/found_diffs.txt', 'a', encoding='utf-8') as file:
                        file.write(f"File: {transcription_txt_file.name}, Trial: {trials - 1}, Different elements: {sym_diff}\n")
                    sym_diff.clear()
            
            if set(unique_entities_names) != set(llm_entity_names) and trials > 1:
                logger = logging.getLogger()
                logger.addHandler(error_file_handler)
                logging.error(f"Failed to get the same entities from the LLM for {transcription_txt_file.name} after {trials - 1} trials.")
                logger.removeHandler(error_file_handler)
                # Continue with what we have
            
            if parsed_llm_entities:
                # Save the LLM output as JSON
                output_llm_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_llm_entities.json"
                save_json_entities(parsed_llm_entities, output_llm_entities_file)
                logging.info(f"LLM entities saved as: {output_llm_entities_file}")

                # Initialize the list of historical entities with the real names and addresses
                expanded_entities = initialize_historical_entities(parsed_llm_entities, unique_entities)
            
                # Now perform entity linking only on real entities
                if not historical_df.empty and expanded_entities:
                    logging.info(f"Performing entity linking on {len(expanded_entities)} expanded entities.")
                    final_entities = entity_linking(expanded_entities, historical_df, entity_linking_threshold)
                    logging.info(f"Entity linking completed for {transcription_txt_file.name}")
                
                # Save the final entities as JSON
                output_final_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_final_entities.json"
                save_json_entities(final_entities, output_final_entities_file)
                logging.info(f"Final entities saved as: {output_final_entities_file}")

            else:
                logging.warning(f"No valid entities found by LLM for {transcription_txt_file.name}")
                # Save empty files to indicate processing was attempted
                
                with open(OUTPUT_DIR / f"{transcription_txt_file.stem}_final_entities.json", "w") as f:
                    f.write("[]")
            
        except Exception as e:
            logger = logging.getLogger()
            logger.addHandler(error_file_handler)
            logging.error(f"Error processing file {transcription_txt_file.name}: {e}")
            logger.removeHandler(error_file_handler)
            continue
        

    # Delete the pipeline
    gc.collect()

    logging.info("NER and filtering workflow completed.")

if __name__ == "__main__":
    main()