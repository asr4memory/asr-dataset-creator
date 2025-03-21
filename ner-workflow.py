from gliner import GLiNER
from pathlib import Path 
import logging
import warnings
from tqdm import tqdm
import json
import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
import torch
from app_config import get_config
import gc
from utils import set_up_logging

def load_gliner_model():
    """
    Load the GLiNER model
    """
    # model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

    return model

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
    logging.info(f"Entities saved as: {output_entities_file}")


def load_llm_model():
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

# def load_model():
#     """
#     Load the LLM model in 8-bit quantized form.
#     """
#     model_id = "meta-llama/Llama-3.3-70B-Instruct"

#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16
#     )
    
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
#     model = AutoModelForCausalLM.from_pretrained(
#         model_id,
#         quantization_config=quantization_config,
#         device_map="auto",
#     )
    
#     pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#     )
    
#     return pipeline


def filter_historical_entities(llm_pipeline, unique_entities, unique_entities_names, trials, max_new_tokens):
    """
    Filter historical entities using a large language model
    """
    sym_diff = set()
    # Define the prompt for the LLM
    prompt_system = (
        "Du erhältst eine Liste von Entitäten (Personen und Adressen), die per Named Entity Recognition erkannt wurden. "
        "1) Bitte prüfe, bei welchen der genannten Entitäten es sich um historische Persönlichkeiten oder Orte handelt, und gib die Antwort ausschließlich im JSON-Format zurück. "
        "2) Bitte prüfe, bei welchen Personen (Entitätentyp 'person') es sich um ausschließlich richtige Vornamen und/oder Nachnamen handelt. Dabei kann es sich entweder nur um einen Vornamen (Beispiel: Max) oder nur um einen Nachnamen (Beispiel: Müller) handeln oder der Vor- und Nachname zusammengeschrieben sein (Beispiel: Max Müller). Die Vor- und Nachnamen können auch in Zusammenhang mit anderen Begriffen stehen (Beispiele: Herr Müller, Frau Müller). Bei dem Entitätentyp 'full address' sollte in diesem Fall der Wert immer 'false' sein. "
        "3) Bitte prüfe, bei welchen Adressen (Entitätentyp 'full address') es sich um richtige Adressen handelt. Bei einer richtigen Adresse handelt es sich ausschließlich nur dann, wenn eine Straße zusammen mit einer Hausnummer auftritt (Beispiel: Musterstraße 1). Bei dem Entitätentyp 'person' sollte in diesem Fall der Wert immer 'false' sein."
        "Jede Entität sollte als ein Objekt im folgenden Format dargestellt werden:\n"
        "{\n"
        '  "entity_name": "<Name der Entität>",\n'
        '  "entity_type": "<Typ der Entität>",\n'
        '  "is_historical": true/false\n'
        '  "is_real_name": true/false\n'
        '  "is_real_address": true/false\n'
        "}\n"
        "Gib ausschließlich gültiges JSON zurück. Keine zusätzlichen Kommentare oder Erklärungen.\n"
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
    logging.info(f"Starting the filtering process with the LLM. Trial: {trials}")
    outputs = llm_pipeline(messages, max_new_tokens=max_new_tokens)
    output = outputs[0]["generated_text"]

    # Extract the assistant content from the output
    assistant_dict = output[2]
    assistant_content = assistant_dict["content"]

    # Parse the assistant content as JSON
    try:
        parsed_llm_entities = json.loads(assistant_content)
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
            # Add entities with is_real_name: false to blacklist
            with open('./data/blacklist.txt', 'a', encoding='utf-8') as file:
                for entity in parsed_llm_entities:
                    if (entity["is_real_name"] == False) and (entity["entity_type"] == "person"):
                        file.write(f"{entity['entity_name'].lower()}\n") 
        else:
            logging.warning("Warning: The gliner and LLM outputs do not contain the same entites. Retrying the LLM filtering process.")
            sym_diff = set(llm_entity_names).symmetric_difference(set(unique_entities_names))
            logging.warning(f"Different elements: {sym_diff}")

    return parsed_llm_entities, llm_entity_names, sym_diff


def save_historical_entities(parsed_llm_entities, transcription_txt_file, OUTPUT_DIR):
    # Save the historical entities as JSON
    output_historical_entities_file = OUTPUT_DIR / f"{transcription_txt_file.stem}_historical_entities.json"
    with open(output_historical_entities_file, "w", encoding="utf-8") as f:
        json.dump(parsed_llm_entities, f, ensure_ascii=False, indent=2)
    logging.info(f"Historical entities saved as: {output_historical_entities_file}")


def main():
    """
    Main function to execute the NER and filtering workflow.
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
    llm_pipeline = load_llm_model()

    # Start the workflow for NER and filtering historical entities
    for transcription_txt_file in tqdm(transcription_txt_files, desc="Processed transcription files", unit="file"):
        try:
            # Start workflow for NER
            logging.info(f"Start processing {transcription_txt_file.name}")
            unique_entities, unique_entities_names = ner_workflow(transcription_txt_file, gliner_model, threshold, batch_size)
        
            # Output entities as tab-separated CSV        
            save_entities_csv(unique_entities, transcription_txt_file, batch_size, OUTPUT_DIR)
            
            # Start workflow for filtering historical entities via LLMs
            trials = 1
            llm_entity_names = []
            while set(unique_entities_names) != set(llm_entity_names) and trials <= 1:
                parsed_llm_entities, llm_entity_names, sym_diff = filter_historical_entities(llm_pipeline, unique_entities, unique_entities_names, trials, max_new_tokens)
                trials += 1
                if sym_diff:
                    with open('./data/found_diffs.txt', 'a', encoding='utf-8') as file:
                        file.write(f"File: {transcription_txt_file.name}, Trial: {trials}, Different elements: {sym_diff}\n")
                    sym_diff.clear()
            
            if set(unique_entities_names) != set(llm_entity_names) and trials == 2:
                logger = logging.getLogger()
                logger.addHandler(error_file_handler)
                logging.error(f"Failed to get the same entities from the LLM for {transcription_txt_file.name} after {trials} trials.")
                logger.removeHandler(error_file_handler)
                continue
            
            # Save the historical entities as JSON
            save_historical_entities(parsed_llm_entities, transcription_txt_file, OUTPUT_DIR)
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