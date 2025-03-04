"""
Default configuration.
These settings are overridden by the settings in the config.toml file,
if present.
"""

CONST_DEFAULT_CONFIG = {
    "logging": {
        "logging_directory": "",
    },    
    "vtt_to_txt": {
        "input_directory": "",
        "output_directory": "",
    },
    "ner_workflow": {
        "input_directory": "",
        "output_directory": "",
        "ner_batch_size": 15,
        "ner_threshold": 0.5,
        "llm_max_new_tokens": 10000,  
    },
    "vtt_anonymization": {
        "input_directory_vtt": "",
        "input_directory_json": "",
        "output_directory": "",
        "vtt_replacements": [], 
    },
    "dataset_creator": {
        "input_directory_vtt": "",
        "input_directory_wav": "",
        "output_directory": "",
        "sample_rate": "16000",
        "offset": 0.075,
    },
    "dataset_merger": {
        "input_directory": "",
        "output_directory": "",
    },
    "hdf5_converter": {
        "input_directory": "",
        "output_directory": "",
        "max_length_samples": 100000,
        "sample_rate": 16000,
        "batch_size": 100,
        "threads": 4,
        "model": "openai/whisper-large-v3",
        "language": "german",
    },
    "dataset_test": {
        "input_directory": "",
    },
}
